#include "splat_constructor/constructor.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace splat {
namespace {

constexpr float kShC0 = 0.28209479177387814F;

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
    if (config.depth_semantics == DepthSemantics::MetricCameraZ) {
        throw std::invalid_argument("8-bit metric camera-Z depth is not supported by this prototype");
    }
}

float logit(float probability) {
    const float bounded = std::clamp(probability, 1.0e-6F, 1.0F - 1.0e-6F);
    return std::log(bounded / (1.0F - bounded));
}

} // namespace

void GaussianSet::validate() const {
    const auto count = size();
    if (positions.size() != count * 3 || sh_dc.size() != count * 3
        || log_scales.size() != count * 3 || rotations.size() != count * 4
        || source_pixels.size() != count * 2) {
        throw std::logic_error("GaussianSet attribute arrays have inconsistent lengths");
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

GaussianSet construct_gaussians(
    const ImageU8& rgb,
    const ImageU8& relative_depth,
    const ImageU8& foreground_mask,
    const CameraIntrinsics& intrinsics,
    const ConstructionConfig& config
) {
    validate_inputs(rgb, relative_depth, foreground_mask, intrinsics, config);

    GaussianSet output;
    const std::size_t approximate_count =
        ((rgb.width + config.sample_stride - 1) / config.sample_stride)
        * ((rgb.height + config.sample_stride - 1) / config.sample_stride);
    output.reserve(approximate_count);

    for (std::size_t v = 0; v < rgb.height; v += config.sample_stride) {
        for (std::size_t u = 0; u < rgb.width; u += config.sample_stride) {
            if (foreground_mask.at(u, v) < config.mask_threshold) {
                continue;
            }

            const float inverse_depth = static_cast<float>(relative_depth.at(u, v)) / 255.0F;
            const float distance = config.near_depth
                + (1.0F - inverse_depth) * (config.far_depth - config.near_depth);
            const float x = (static_cast<float>(u) - intrinsics.cx) * distance / intrinsics.fx;
            const float y = -(static_cast<float>(v) - intrinsics.cy) * distance / intrinsics.fy;
            const float z = -distance;
            output.positions.insert(output.positions.end(), {x, y, z});

            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float color = static_cast<float>(rgb.at(u, v, channel)) / 255.0F;
                output.sh_dc.push_back((color - 0.5F) / kShC0);
            }

            output.opacity_logits.push_back(logit(config.opacity));
            const float pixel_footprint = distance / intrinsics.fx
                * static_cast<float>(config.sample_stride) * config.scale_multiplier;
            const float tangential_log_scale = std::log(std::max(pixel_footprint, 1.0e-6F));
            const float normal_log_scale = std::log(
                std::max(pixel_footprint * config.thickness_multiplier, 1.0e-6F)
            );
            output.log_scales.insert(
                output.log_scales.end(),
                {tangential_log_scale, tangential_log_scale, normal_log_scale}
            );
            output.rotations.insert(output.rotations.end(), {1.0F, 0.0F, 0.0F, 0.0F});
            output.source_pixels.push_back(static_cast<std::uint32_t>(u));
            output.source_pixels.push_back(static_cast<std::uint32_t>(v));
        }
    }

    output.validate();
    return output;
}

} // namespace splat
