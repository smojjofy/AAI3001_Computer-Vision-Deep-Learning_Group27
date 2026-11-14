#include "splat_constructor/constructor.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace splat {
namespace {

constexpr float kShC0 = 0.28209479177387814F;

struct Vec3 {
    float x{};
    float y{};
    float z{};
};

Vec3 subtract(const Vec3& left, const Vec3& right) {
    return Vec3{left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 multiply(const Vec3& value, float scalar) {
    return Vec3{value.x * scalar, value.y * scalar, value.z * scalar};
}

float dot(const Vec3& left, const Vec3& right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vec3 cross(const Vec3& left, const Vec3& right) {
    return Vec3{
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

Vec3 normalize(const Vec3& value, const Vec3& fallback) {
    const float length = std::sqrt(dot(value, value));
    return length > 1.0e-8F ? multiply(value, 1.0F / length) : fallback;
}

void validate_config(const ConstructionConfig& config) {
    if (config.sample_stride == 0) {
        throw std::invalid_argument("sample stride must be at least one");
    }
    if (!(config.near_depth > 0.0F) || !(config.far_depth > config.near_depth)) {
        throw std::invalid_argument("depth range must satisfy 0 < near < far");
    }
    if (!(config.opacity > 0.0F && config.opacity < 1.0F)) {
        throw std::invalid_argument("opacity must be strictly between zero and one");
    }
    if (!(config.scale_multiplier > 0.0F) || !(config.thickness_multiplier > 0.0F)) {
        throw std::invalid_argument("scale multipliers must be positive");
    }
    if (!std::isfinite(config.depth_edge_threshold) || config.depth_edge_threshold < 0.0F
        || !(config.edge_scale_multiplier > 0.0F && config.edge_scale_multiplier <= 1.0F)
        || !std::isfinite(config.minimum_reconstruction_weight)
        || config.minimum_reconstruction_weight < 0.0F
        || config.minimum_reconstruction_weight >= 1.0F) {
        throw std::invalid_argument("invalid edge or reconstruction-weight configuration");
    }
}

void validate_inputs(
    const ImageU8& rgb,
    const ImageU8& depth,
    const ImageU8& mask,
    const CameraIntrinsics& intrinsics,
    const ConstructionConfig& config
) {
    if (rgb.channels != 3 || depth.channels != 1 || mask.channels != 1) {
        throw std::invalid_argument("expected RGB PPM plus single-channel depth and mask PGM");
    }
    if (rgb.width == 0 || rgb.height == 0 || depth.width != rgb.width
        || depth.height != rgb.height || mask.width != rgb.width || mask.height != rgb.height) {
        throw std::invalid_argument("RGB, depth, and mask dimensions must match and be non-zero");
    }
    if (rgb.pixels.size() != rgb.width * rgb.height * 3
        || depth.pixels.size() != depth.width * depth.height
        || mask.pixels.size() != mask.width * mask.height) {
        throw std::invalid_argument("image payload size does not match its dimensions");
    }
    if (!(intrinsics.fx > 0.0F) || !(intrinsics.fy > 0.0F)) {
        throw std::invalid_argument("camera focal lengths must be positive");
    }
    validate_config(config);
    if (config.depth_semantics == DepthSemantics::MetricCameraZ) {
        throw std::invalid_argument("8-bit metric camera-Z depth is not supported by this prototype");
    }
}

float logit(float probability) {
    const float bounded = std::clamp(probability, 1.0e-6F, 1.0F - 1.0e-6F);
    return std::log(bounded / (1.0F - bounded));
}

float reconstruction_weight(const ConstructionFrame& frame, std::size_t index) {
    const float validity = frame.depth_validity.pixels.empty()
        ? 1.0F : frame.depth_validity.pixels[index];
    const float foreground = frame.foreground_weight.pixels.empty()
        ? 1.0F : frame.foreground_weight.pixels[index];
    return validity * foreground;
}

float distance_at(const ConstructionFrame& frame, const ConstructionConfig& config, std::size_t index) {
    const float depth = frame.depth.pixels[index];
    if (frame.depth_semantics == DepthSemantics::MetricCameraZ) {
        return depth;
    }
    return config.near_depth + (1.0F - depth) * (config.far_depth - config.near_depth);
}

Vec3 camera_position(
    std::size_t u,
    std::size_t v,
    float distance,
    const CameraIntrinsics& intrinsics
) {
    return Vec3{
        (static_cast<float>(u) - intrinsics.cx) * distance / intrinsics.fx,
        -(static_cast<float>(v) - intrinsics.cy) * distance / intrinsics.fy,
        -distance,
    };
}

Vec3 transform_point(const std::array<float, 16>& matrix, const Vec3& point) {
    return Vec3{
        matrix[0] * point.x + matrix[1] * point.y + matrix[2] * point.z + matrix[3],
        matrix[4] * point.x + matrix[5] * point.y + matrix[6] * point.z + matrix[7],
        matrix[8] * point.x + matrix[9] * point.y + matrix[10] * point.z + matrix[11],
    };
}

Vec3 transform_direction(const std::array<float, 16>& matrix, const Vec3& direction) {
    return Vec3{
        matrix[0] * direction.x + matrix[1] * direction.y + matrix[2] * direction.z,
        matrix[4] * direction.x + matrix[5] * direction.y + matrix[6] * direction.z,
        matrix[8] * direction.x + matrix[9] * direction.y + matrix[10] * direction.z,
    };
}

std::array<float, 4> quaternion_from_basis(Vec3 axis_x, Vec3 axis_y, Vec3 axis_z) {
    axis_x = normalize(axis_x, Vec3{1.0F, 0.0F, 0.0F});
    axis_y = normalize(
        subtract(axis_y, multiply(axis_x, dot(axis_x, axis_y))),
        Vec3{0.0F, 1.0F, 0.0F}
    );
    axis_z = normalize(cross(axis_x, axis_y), axis_z);

    const float m00 = axis_x.x;
    const float m01 = axis_y.x;
    const float m02 = axis_z.x;
    const float m10 = axis_x.y;
    const float m11 = axis_y.y;
    const float m12 = axis_z.y;
    const float m20 = axis_x.z;
    const float m21 = axis_y.z;
    const float m22 = axis_z.z;
    const float trace = m00 + m11 + m22;
    std::array<float, 4> quaternion{};
    if (trace > 0.0F) {
        const float scale = 2.0F * std::sqrt(trace + 1.0F);
        quaternion = {0.25F * scale, (m21 - m12) / scale, (m02 - m20) / scale,
                      (m10 - m01) / scale};
    } else if (m00 > m11 && m00 > m22) {
        const float scale = 2.0F * std::sqrt(1.0F + m00 - m11 - m22);
        quaternion = {(m21 - m12) / scale, 0.25F * scale, (m01 + m10) / scale,
                      (m02 + m20) / scale};
    } else if (m11 > m22) {
        const float scale = 2.0F * std::sqrt(1.0F + m11 - m00 - m22);
        quaternion = {(m02 - m20) / scale, (m01 + m10) / scale, 0.25F * scale,
                      (m12 + m21) / scale};
    } else {
        const float scale = 2.0F * std::sqrt(1.0F + m22 - m00 - m11);
        quaternion = {(m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale,
                      0.25F * scale};
    }
    const float norm = std::sqrt(
        quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]
    );
    for (float& component : quaternion) {
        component /= norm;
    }
    return quaternion;
}

bool usable_pixel(
    const ConstructionFrame& frame,
    const ConstructionConfig& config,
    std::size_t u,
    std::size_t v
) {
    if (u >= frame.rgb.width || v >= frame.rgb.height) {
        return false;
    }
    const std::size_t index = v * frame.rgb.width + u;
    if (frame.foreground_mask.pixels[index] == 0
        || reconstruction_weight(frame, index) <= config.minimum_reconstruction_weight) {
        return false;
    }
    return frame.depth_semantics != DepthSemantics::MetricCameraZ
        || frame.depth.pixels[index] > 0.0F;
}

bool continuous_neighbor(
    const ConstructionFrame& frame,
    const ConstructionConfig& config,
    std::size_t center_u,
    std::size_t center_v,
    std::size_t neighbor_u,
    std::size_t neighbor_v
) {
    if (!usable_pixel(frame, config, neighbor_u, neighbor_v)) {
        return false;
    }
    const float center = distance_at(frame, config, center_v * frame.rgb.width + center_u);
    const float neighbor = distance_at(frame, config, neighbor_v * frame.rgb.width + neighbor_u);
    const float denominator = std::max(std::max(center, neighbor), 1.0e-6F);
    return std::abs(center - neighbor) / denominator <= config.depth_edge_threshold;
}

} // namespace

void GaussianSet::validate() const {
    const auto count = size();
    if (positions.size() != count * 3 || sh_dc.size() != count * 3
        || log_scales.size() != count * 3 || rotations.size() != count * 4
        || reconstruction_weights.size() != count || local_ids.size() != count
        || source_pixels.size() != count * 2) {
        throw std::logic_error("GaussianSet attribute arrays have inconsistent lengths");
    }
    const auto all_finite = [](const std::vector<float>& values) {
        return std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
    };
    if (!all_finite(positions) || !all_finite(sh_dc) || !all_finite(opacity_logits)
        || !all_finite(log_scales) || !all_finite(rotations)
        || !std::ranges::all_of(reconstruction_weights, [](float value) {
               return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
           })) {
        throw std::logic_error("GaussianSet attributes must be finite and weights must be in [0, 1]");
    }
    for (std::size_t index = 0; index < count; ++index) {
        const float w = rotations[index * 4];
        const float x = rotations[index * 4 + 1];
        const float y = rotations[index * 4 + 2];
        const float z = rotations[index * 4 + 3];
        const float norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (std::abs(norm - 1.0F) > 1.0e-4F) {
            throw std::logic_error("GaussianSet rotations must be unit quaternions");
        }
    }
}

CameraIntrinsics intrinsics_from_horizontal_fov(
    std::size_t width,
    std::size_t height,
    float horizontal_fov_degrees
) {
    if (width == 0 || height == 0 || !(horizontal_fov_degrees > 0.0F)
        || !(horizontal_fov_degrees < 180.0F)) {
        throw std::invalid_argument("invalid dimensions or horizontal field of view");
    }
    const float radians = horizontal_fov_degrees * std::numbers::pi_v<float> / 180.0F;
    const float focal = static_cast<float>(width) / (2.0F * std::tan(radians / 2.0F));
    return CameraIntrinsics{
        focal,
        focal,
        (static_cast<float>(width) - 1.0F) / 2.0F,
        (static_cast<float>(height) - 1.0F) / 2.0F,
    };
}

SplatState construct_splat_state(
    const ConstructionFrame& frame,
    const ConstructionConfig& config
) {
    frame.validate();
    validate_config(config);

    SplatState state;
    state.frame_id = frame.frame_id;
    state.timestamp_ns = frame.timestamp_ns;
    state.source_id = frame.source_id;
    state.source_width = frame.rgb.width;
    state.source_height = frame.rgb.height;
    state.intrinsics = frame.intrinsics;
    state.camera_to_world = frame.camera_to_world;
    state.source_depth_semantics = frame.depth_semantics;
    state.source_depth_units = frame.depth_units;

    GaussianSet& output = state.gaussians;
    const std::size_t approximate_count =
        ((frame.rgb.width + config.sample_stride - 1) / config.sample_stride)
        * ((frame.rgb.height + config.sample_stride - 1) / config.sample_stride);
    output.reserve(approximate_count);

    for (std::size_t v = 0; v < frame.rgb.height; v += config.sample_stride) {
        for (std::size_t u = 0; u < frame.rgb.width; u += config.sample_stride) {
            if (!usable_pixel(frame, config, u, v)) {
                continue;
            }

            const std::size_t pixel_index = v * frame.rgb.width + u;
            const float distance = distance_at(frame, config, pixel_index);
            const Vec3 camera_point = camera_position(u, v, distance, frame.intrinsics);
            const Vec3 world_point = transform_point(frame.camera_to_world, camera_point);
            output.positions.insert(
                output.positions.end(), {world_point.x, world_point.y, world_point.z}
            );

            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float color = static_cast<float>(frame.rgb.at(u, v, channel)) / 255.0F;
                output.sh_dc.push_back((color - 0.5F) / kShC0);
            }

            const float weight = reconstruction_weight(frame, pixel_index);
            output.opacity_logits.push_back(logit(config.opacity * weight));

            const bool has_left = u > 0
                && continuous_neighbor(frame, config, u, v, u - 1, v);
            const bool has_right = u + 1 < frame.rgb.width
                && continuous_neighbor(frame, config, u, v, u + 1, v);
            const bool has_up = v > 0
                && continuous_neighbor(frame, config, u, v, u, v - 1);
            const bool has_down = v + 1 < frame.rgb.height
                && continuous_neighbor(frame, config, u, v, u, v + 1);

            float scale_x = distance / frame.intrinsics.fx
                * static_cast<float>(config.sample_stride) * config.scale_multiplier;
            float scale_y = distance / frame.intrinsics.fy
                * static_cast<float>(config.sample_stride) * config.scale_multiplier;
            if (!(has_left && has_right)) {
                scale_x *= config.edge_scale_multiplier;
            }
            if (!(has_up && has_down)) {
                scale_y *= config.edge_scale_multiplier;
            }
            const float scale_z = std::sqrt(scale_x * scale_y) * config.thickness_multiplier;
            output.log_scales.insert(
                output.log_scales.end(),
                {
                    std::log(std::max(scale_x, 1.0e-6F)),
                    std::log(std::max(scale_y, 1.0e-6F)),
                    std::log(std::max(scale_z, 1.0e-6F)),
                }
            );

            Vec3 axis_x{1.0F, 0.0F, 0.0F};
            Vec3 axis_y{0.0F, 1.0F, 0.0F};
            Vec3 axis_z{0.0F, 0.0F, 1.0F};
            if (has_left && has_right && has_up && has_down) {
                const float left_distance = distance_at(frame, config, pixel_index - 1);
                const float right_distance = distance_at(frame, config, pixel_index + 1);
                const float up_distance = distance_at(
                    frame, config, pixel_index - frame.rgb.width
                );
                const float down_distance = distance_at(
                    frame, config, pixel_index + frame.rgb.width
                );
                axis_x = normalize(
                    subtract(
                        camera_position(u + 1, v, right_distance, frame.intrinsics),
                        camera_position(u - 1, v, left_distance, frame.intrinsics)
                    ),
                    axis_x
                );
                axis_y = normalize(
                    subtract(
                        camera_position(u, v - 1, up_distance, frame.intrinsics),
                        camera_position(u, v + 1, down_distance, frame.intrinsics)
                    ),
                    axis_y
                );
                axis_y = normalize(
                    subtract(axis_y, multiply(axis_x, dot(axis_x, axis_y))), axis_y
                );
                axis_z = normalize(cross(axis_x, axis_y), axis_z);
            }
            axis_x = transform_direction(frame.camera_to_world, axis_x);
            axis_y = transform_direction(frame.camera_to_world, axis_y);
            axis_z = transform_direction(frame.camera_to_world, axis_z);
            const auto rotation = quaternion_from_basis(axis_x, axis_y, axis_z);
            output.rotations.insert(output.rotations.end(), rotation.begin(), rotation.end());
            output.reconstruction_weights.push_back(weight);
            output.local_ids.push_back(
                static_cast<std::uint64_t>(v) * static_cast<std::uint64_t>(frame.rgb.width)
                + static_cast<std::uint64_t>(u)
            );
            output.source_pixels.push_back(static_cast<std::uint32_t>(u));
            output.source_pixels.push_back(static_cast<std::uint32_t>(v));
        }
    }

    state.validate();
    return state;
}

GaussianSet construct_gaussians(
    const ImageU8& rgb,
    const ImageU8& relative_depth,
    const ImageU8& foreground_mask,
    const CameraIntrinsics& intrinsics,
    const ConstructionConfig& config
) {
    validate_inputs(rgb, relative_depth, foreground_mask, intrinsics, config);

    ConstructionFrame frame;
    frame.frame_id = "legacy_fixture";
    frame.timestamp_ns = 1;
    frame.source_id = "legacy_cli";
    frame.rgb = rgb;
    frame.depth = ImageF32{rgb.width, rgb.height, 1, {}};
    frame.depth.pixels.reserve(relative_depth.pixels.size());
    for (const std::uint8_t value : relative_depth.pixels) {
        frame.depth.pixels.push_back(static_cast<float>(value) / 255.0F);
    }
    frame.foreground_mask = ImageU8{rgb.width, rgb.height, 1, {}};
    frame.foreground_mask.pixels.reserve(foreground_mask.pixels.size());
    for (const std::uint8_t value : foreground_mask.pixels) {
        frame.foreground_mask.pixels.push_back(value >= config.mask_threshold ? 1 : 0);
    }
    frame.intrinsics = intrinsics;
    frame.depth_semantics = config.depth_semantics;
    frame.depth_units = DepthUnits::Unitless;
    return construct_splat_state(frame, config).gaussians;
}

} // namespace splat
