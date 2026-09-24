#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "splat_constructor/constructor.hpp"
#include "splat_constructor/ply_writer.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(float left, float right, float tolerance = 1.0e-5F) {
    return std::abs(left - right) <= tolerance;
}

splat::ImageU8 image(
    std::size_t width,
    std::size_t height,
    std::size_t channels,
    std::uint8_t value
) {
    return splat::ImageU8{
        width,
        height,
        channels,
        std::vector<std::uint8_t>(width * height * channels, value),
    };
}

void test_center_projection_and_attributes() {
    auto rgb = image(3, 3, 3, 0);
    rgb.pixels[(1 * 3 + 1) * 3] = 255;
    auto depth = image(3, 3, 1, 255);
    auto mask = image(3, 3, 1, 0);
    mask.pixels[1 * 3 + 1] = 255;

    splat::ConstructionConfig config;
    config.sample_stride = 1;
    config.near_depth = 1.0F;
    config.far_depth = 4.0F;
    const splat::CameraIntrinsics intrinsics{2.0F, 2.0F, 1.0F, 1.0F};
    const auto gaussians = splat::construct_gaussians(rgb, depth, mask, intrinsics, config);

    require(gaussians.size() == 1, "expected exactly one masked Gaussian");
    require(close(gaussians.positions[0], 0.0F), "center x must be zero");
    require(close(gaussians.positions[1], 0.0F), "center y must be zero");
    require(close(gaussians.positions[2], -1.0F), "bright inverse depth must map to near -Z");
    require(gaussians.sh_dc[0] > 0.0F, "red SH coefficient must be positive");
    require(gaussians.sh_dc[1] < 0.0F && gaussians.sh_dc[2] < 0.0F,
            "zero green and blue must produce negative SH coefficients");
    require(close(gaussians.rotations[0], 1.0F), "identity quaternion must use wxyz order");
    require(gaussians.source_pixels[0] == 1 && gaussians.source_pixels[1] == 1,
            "source pixel association must be preserved");
}

void test_empty_mask() {
    const auto rgb = image(4, 2, 3, 127);
    const auto depth = image(4, 2, 1, 127);
    const auto mask = image(4, 2, 1, 0);
    const auto intrinsics = splat::intrinsics_from_horizontal_fov(4, 2, 50.0F);
    const auto gaussians = splat::construct_gaussians(
        rgb, depth, mask, intrinsics, splat::ConstructionConfig{}
    );
    require(gaussians.size() == 0, "empty mask must produce no Gaussians");
}

void test_reference_compatible_ply_header() {
    auto rgb = image(1, 1, 3, 127);
    auto depth = image(1, 1, 1, 255);
    auto mask = image(1, 1, 1, 255);
    const splat::CameraIntrinsics intrinsics{1.0F, 1.0F, 0.0F, 0.0F};
    const auto gaussians = splat::construct_gaussians(
        rgb, depth, mask, intrinsics, splat::ConstructionConfig{}
    );

    const auto path = std::filesystem::temp_directory_path() / "group27_splat_test.ply";
    splat::write_3dgs_ply(path, gaussians);
    std::ifstream input(path, std::ios::binary);
    std::string header;
    std::string line;
    while (std::getline(input, line)) {
        header += line + '\n';
        if (line == "end_header") {
            break;
        }
    }
    input.close();
    std::filesystem::remove(path);

    require(header.find("format binary_little_endian 1.0") != std::string::npos,
            "PLY must be binary little endian");
    require(header.find("element vertex 1") != std::string::npos,
            "PLY vertex count must match Gaussian count");
    require(header.find("property float f_rest_44") != std::string::npos,
            "PLY must contain the reference degree-3 SH schema");
    require(header.find("property float rot_3") != std::string::npos,
            "PLY must contain the complete quaternion");
}

} // namespace

int main() {
    try {
        test_center_projection_and_attributes();
        test_empty_mask();
        test_reference_compatible_ply_header();
        std::cout << "All Splat Constructor tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
