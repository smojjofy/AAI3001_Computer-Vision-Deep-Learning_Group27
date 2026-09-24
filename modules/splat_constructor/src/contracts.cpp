#include "splat_constructor/types.hpp"

#include <cmath>
#include <stdexcept>

namespace splat {
namespace {

void validate_transform(const std::array<float, 16>& transform, const char* owner) {
    for (const float value : transform) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(owner) + " camera_to_world values must be finite");
        }
    }
    constexpr float tolerance = 1.0e-5F;
    if (std::abs(transform[12]) > tolerance || std::abs(transform[13]) > tolerance
        || std::abs(transform[14]) > tolerance || std::abs(transform[15] - 1.0F) > tolerance) {
        throw std::invalid_argument(std::string(owner) + " camera_to_world must be row-major affine");
    }
    const std::array<float, 3> axis_x{transform[0], transform[4], transform[8]};
    const std::array<float, 3> axis_y{transform[1], transform[5], transform[9]};
    const std::array<float, 3> axis_z{transform[2], transform[6], transform[10]};
    const auto dot = [](const auto& left, const auto& right) {
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    };
    const float determinant =
        axis_x[0] * (axis_y[1] * axis_z[2] - axis_y[2] * axis_z[1])
        - axis_y[0] * (axis_x[1] * axis_z[2] - axis_x[2] * axis_z[1])
        + axis_z[0] * (axis_x[1] * axis_y[2] - axis_x[2] * axis_y[1]);
    constexpr float rigid_tolerance = 1.0e-3F;
    if (std::abs(dot(axis_x, axis_x) - 1.0F) > rigid_tolerance
        || std::abs(dot(axis_y, axis_y) - 1.0F) > rigid_tolerance
        || std::abs(dot(axis_z, axis_z) - 1.0F) > rigid_tolerance
        || std::abs(dot(axis_x, axis_y)) > rigid_tolerance
        || std::abs(dot(axis_x, axis_z)) > rigid_tolerance
        || std::abs(dot(axis_y, axis_z)) > rigid_tolerance
        || std::abs(determinant - 1.0F) > rigid_tolerance) {
        throw std::invalid_argument(
            std::string(owner) + " camera_to_world rotation must be right-handed and rigid"
        );
    }
}

void validate_optional_plane(
    const ImageF32& image,
    std::size_t width,
    std::size_t height,
    const char* name
) {
    if (image.pixels.empty()) {
        if (image.width != 0 || image.height != 0 || image.channels != 0) {
            throw std::invalid_argument(std::string(name) + " must be wholly absent or populated");
        }
        return;
    }
    if (image.width != width || image.height != height || image.channels != 1
        || image.pixels.size() != width * height) {
        throw std::invalid_argument(std::string(name) + " must be an aligned single-channel image");
    }
    for (const float value : image.pixels) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
            throw std::invalid_argument(std::string(name) + " values must be finite in [0, 1]");
        }
    }
}

} // namespace

void ConstructionFrame::validate() const {
    if (frame_id.empty() || source_id.empty()) {
        throw std::invalid_argument("frame_id and source_id must not be empty");
    }
    if (timestamp_ns == 0) {
        throw std::invalid_argument("timestamp_ns must be a positive monotonic timestamp");
    }
    if (rgb.width == 0 || rgb.height == 0 || rgb.channels != 3
        || rgb.pixels.size() != rgb.width * rgb.height * 3) {
        throw std::invalid_argument("rgb must be a non-empty, packed three-channel image");
    }
    const auto pixel_count = rgb.width * rgb.height;
    if (depth.width != rgb.width || depth.height != rgb.height || depth.channels != 1
        || depth.pixels.size() != pixel_count) {
        throw std::invalid_argument("depth must be an aligned single-channel float image");
    }
    if (foreground_mask.width != rgb.width || foreground_mask.height != rgb.height
        || foreground_mask.channels != 1 || foreground_mask.pixels.size() != pixel_count) {
        throw std::invalid_argument("foreground_mask must be an aligned single-channel image");
    }
    for (const float value : depth.pixels) {
        if (!std::isfinite(value) || value < 0.0F
            || (depth_semantics != DepthSemantics::MetricCameraZ && value > 1.0F)) {
            throw std::invalid_argument(
                "depth must be non-negative and relative depth must remain in [0, 1]"
            );
        }
    }
    for (const std::uint8_t value : foreground_mask.pixels) {
        if (value != 0 && value != 1) {
            throw std::invalid_argument("foreground_mask values must be binary {0, 1}");
        }
    }
    validate_optional_plane(depth_validity, rgb.width, rgb.height, "depth_validity");
    validate_optional_plane(foreground_weight, rgb.width, rgb.height, "foreground_weight");
    if (!(intrinsics.fx > 0.0F) || !(intrinsics.fy > 0.0F)
        || !std::isfinite(intrinsics.cx) || !std::isfinite(intrinsics.cy)) {
        throw std::invalid_argument("camera intrinsics must be finite with positive focal lengths");
    }
    validate_transform(camera_to_world, "ConstructionFrame");
    if (depth_semantics == DepthSemantics::MetricCameraZ && depth_units != DepthUnits::Meters) {
        throw std::invalid_argument("metric camera-Z depth must use meter units");
    }
    if (depth_semantics != DepthSemantics::MetricCameraZ && depth_units != DepthUnits::Unitless) {
        throw std::invalid_argument("relative depth must use unitless units");
    }
}

void SplatState::validate() const {
    if (frame_id.empty() || source_id.empty() || source_width == 0 || source_height == 0) {
        throw std::invalid_argument("SplatState provenance and source dimensions are required");
    }
    if (!(intrinsics.fx > 0.0F) || !(intrinsics.fy > 0.0F)) {
        throw std::invalid_argument("SplatState focal lengths must be positive");
    }
    if (timestamp_ns == 0) {
        throw std::invalid_argument("SplatState timestamp_ns must be positive");
    }
    validate_transform(camera_to_world, "SplatState");
    if (source_depth_semantics == DepthSemantics::MetricCameraZ
        && source_depth_units != DepthUnits::Meters) {
        throw std::invalid_argument("SplatState metric source depth must use meters");
    }
    if (source_depth_semantics != DepthSemantics::MetricCameraZ
        && source_depth_units != DepthUnits::Unitless) {
        throw std::invalid_argument("SplatState relative source depth must be unitless");
    }
    gaussians.validate();
}

} // namespace splat
