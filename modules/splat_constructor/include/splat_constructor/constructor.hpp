#pragma once

#include "splat_constructor/types.hpp"

namespace splat {

[[nodiscard]] CameraIntrinsics intrinsics_from_horizontal_fov(
    std::size_t width,
    std::size_t height,
    float horizontal_fov_degrees
);

[[nodiscard]] GaussianSet construct_gaussians(
    const ImageU8& rgb,
    const ImageU8& relative_depth,
    const ImageU8& foreground_mask,
    const CameraIntrinsics& intrinsics,
    const ConstructionConfig& config
);

} // namespace splat
