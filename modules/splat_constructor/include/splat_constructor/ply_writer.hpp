#pragma once

#include <filesystem>

#include "splat_constructor/types.hpp"

namespace splat {

void write_3dgs_ply(const std::filesystem::path& path, const GaussianSet& gaussians);

void write_construction_metadata(
    const std::filesystem::path& path,
    const GaussianSet& gaussians,
    const CameraIntrinsics& intrinsics,
    const ConstructionConfig& config,
    std::size_t width,
    std::size_t height,
    float horizontal_fov_degrees
);

} // namespace splat
