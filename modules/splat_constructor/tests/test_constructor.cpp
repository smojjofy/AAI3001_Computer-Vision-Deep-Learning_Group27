#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "splat_constructor/constructor.hpp"
#include "splat_constructor/image_io.hpp"
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

std::filesystem::path temporary_path(const char* filename) {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
        / (std::string("group27_") + std::to_string(suffix) + "_" + filename);
}

template <typename Callable>
void require_throws(Callable callable, const char* message) {
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
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

splat::ConstructionFrame construction_frame(
    std::size_t width,
    std::size_t height,
    const std::vector<float>& depth,
    splat::DepthSemantics semantics = splat::DepthSemantics::RelativeInverse
) {
    splat::ConstructionFrame frame;
    frame.frame_id = "0001";
    frame.timestamp_ns = 123;
    frame.source_id = "0001";
    frame.rgb = image(width, height, 3, 127);
    frame.depth = splat::ImageF32{width, height, 1, depth};
    frame.foreground_mask = image(width, height, 1, 1);
    frame.intrinsics = splat::CameraIntrinsics{2.0F, 2.0F,
        (static_cast<float>(width) - 1.0F) / 2.0F,
        (static_cast<float>(height) - 1.0F) / 2.0F};
    frame.depth_semantics = semantics;
    frame.depth_units = semantics == splat::DepthSemantics::MetricCameraZ
        ? splat::DepthUnits::Meters : splat::DepthUnits::Unitless;
    return frame;
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
    require(gaussians.local_ids[0] == 4, "local ID must equal the source pixel index");
    require(close(gaussians.reconstruction_weights[0], 1.0F),
            "baseline reconstruction weight must be one");
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

void test_reference_compatible_ply_payload() {
    auto rgb = image(1, 1, 3, 127);
    auto depth = image(1, 1, 1, 255);
    auto mask = image(1, 1, 1, 255);
    const splat::CameraIntrinsics intrinsics{1.0F, 1.0F, 0.0F, 0.0F};
    const auto gaussians = splat::construct_gaussians(
        rgb, depth, mask, intrinsics, splat::ConstructionConfig{}
    );

    const auto path = temporary_path("splat_test.ply");
    splat::write_3dgs_ply(path, gaussians);
    // Replacement must also succeed so atomic output supports repeatable runs.
    splat::write_3dgs_ply(path, gaussians);
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "PLY test output must open");
    std::string header;
    std::string line;
    while (std::getline(input, line)) {
        header += line + '\n';
        if (line == "end_header") {
            break;
        }
    }
    require(header.find("format binary_little_endian 1.0") != std::string::npos,
            "PLY must be binary little endian");
    require(header.find("element vertex 1") != std::string::npos,
            "PLY vertex count must match Gaussian count");
    require(header.find("property float f_rest_44") != std::string::npos,
            "PLY must contain the reference degree-3 SH schema");
    require(header.find("property float rot_3") != std::string::npos,
            "PLY must contain the complete quaternion");

    std::array<float, 62> record{};
    input.read(reinterpret_cast<char*>(record.data()), sizeof(record));
    require(input.gcount() == static_cast<std::streamsize>(sizeof(record)),
            "PLY must contain one complete 62-float record");
    require(input.peek() == std::char_traits<char>::eof(),
            "PLY must not contain trailing payload bytes");
    input.close();
    std::filesystem::remove(path);

    require(close(record[0], gaussians.positions[0])
                && close(record[1], gaussians.positions[1])
                && close(record[2], gaussians.positions[2]),
            "PLY position field order is incorrect");
    require(close(record[3], 0.0F) && close(record[5], 0.0F),
            "PLY normals must be zero-filled");
    require(close(record[6], gaussians.sh_dc[0]) && close(record[8], gaussians.sh_dc[2]),
            "PLY SH DC field order is incorrect");
    for (std::size_t index = 9; index < 54; ++index) {
        require(close(record[index], 0.0F), "PLY higher-order SH fields must be zero-filled");
    }
    require(close(record[54], gaussians.opacity_logits[0]), "PLY opacity field is incorrect");
    require(close(record[55], gaussians.log_scales[0])
                && close(record[57], gaussians.log_scales[2]),
            "PLY scale field order is incorrect");
    require(close(record[58], 1.0F) && close(record[59], 0.0F)
                && close(record[60], 0.0F) && close(record[61], 0.0F),
            "PLY quaternion must use wxyz order");
}

void test_portable_image_crlf_and_trailing_data() {
    const auto valid_path = temporary_path("crlf.ppm");
    {
        std::ofstream output(valid_path, std::ios::binary);
        output << "P6\r\n1 1\r\n255\r\n";
        const std::array<unsigned char, 3> pixel{10, 20, 30};
        output.write(reinterpret_cast<const char*>(pixel.data()), pixel.size());
    }
    const auto image = splat::read_portable_image(valid_path);
    std::filesystem::remove(valid_path);
    require(image.width == 1 && image.height == 1 && image.channels == 3,
            "CRLF PPM dimensions must decode");
    require(image.pixels == std::vector<std::uint8_t>({10, 20, 30}),
            "CRLF separator must not shift pixel bytes");

    const auto trailing_path = temporary_path("trailing.pgm");
    {
        std::ofstream output(trailing_path, std::ios::binary);
        output << "P5\n1 1\n255\n";
        const std::array<unsigned char, 2> bytes{42, 99};
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    require_throws(
        [&]() { static_cast<void>(splat::read_portable_image(trailing_path)); },
        "portable image reader must reject trailing bytes"
    );
    std::filesystem::remove(trailing_path);
}

void test_metadata_is_complete_and_replaceable() {
    const auto rgb = image(1, 1, 3, 127);
    const auto depth = image(1, 1, 1, 255);
    const auto mask = image(1, 1, 1, 255);
    const splat::CameraIntrinsics intrinsics{2.0F, 2.0F, 0.0F, 0.0F};
    splat::ConstructionConfig config;
    config.sample_stride = 3;
    config.mask_threshold = 0;
    config.opacity = 0.8F;
    config.scale_multiplier = 0.6F;
    config.thickness_multiplier = 0.2F;
    config.depth_edge_threshold = 0.15F;
    config.edge_scale_multiplier = 0.4F;
    config.minimum_reconstruction_weight = 0.1F;
    const auto gaussians = splat::construct_gaussians(rgb, depth, mask, intrinsics, config);
    const auto path = temporary_path("metadata.json");

    splat::write_construction_metadata(path, gaussians, intrinsics, config, 1, 1, 55.0F);
    splat::write_construction_metadata(path, gaussians, intrinsics, config, 1, 1, 55.0F);
    std::ifstream input(path);
    require(static_cast<bool>(input), "metadata test output must open");
    const std::string contents(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()
    );
    input.close();
    std::filesystem::remove(path);

    require(contents.find("\"mask_threshold\": 0") != std::string::npos,
            "metadata must preserve a zero mask threshold");
    require(contents.find("\"opacity\": 0.8") != std::string::npos,
            "metadata must record opacity");
    require(contents.find("\"scale_multiplier\": 0.6") != std::string::npos,
            "metadata must record scale multiplier");
    require(contents.find("\"thickness_multiplier\": 0.2") != std::string::npos,
            "metadata must record thickness multiplier");
    require(contents.find("\"horizontal_fov_degrees\": 55") != std::string::npos,
            "metadata must record horizontal field of view");
    require(contents.find("\"depth_edge_threshold\": 0.15") != std::string::npos,
            "metadata must record depth edge threshold");
    require(contents.find("\"edge_scale_multiplier\": 0.4") != std::string::npos,
            "metadata must record edge scale multiplier");
    require(contents.find("\"minimum_reconstruction_weight\": 0.1") != std::string::npos,
            "metadata must record reconstruction weight threshold");
}

void test_contract_validation() {
    splat::ConstructionFrame frame;
    frame.frame_id = "0001";
    frame.timestamp_ns = 123;
    frame.source_id = "0001";
    frame.rgb = image(2, 1, 3, 127);
    frame.depth = splat::ImageF32{2, 1, 1, {0.25F, 0.75F}};
    frame.foreground_mask = splat::ImageU8{2, 1, 1, {1, 0}};
    frame.intrinsics = splat::CameraIntrinsics{2.0F, 2.0F, 0.5F, 0.0F};
    frame.validate();

    const auto depth_u8 = image(2, 1, 1, 127);
    const auto mask_u8 = image(2, 1, 1, 255);
    splat::SplatState state;
    state.frame_id = frame.frame_id;
    state.timestamp_ns = frame.timestamp_ns;
    state.source_id = frame.source_id;
    state.source_width = frame.rgb.width;
    state.source_height = frame.rgb.height;
    state.intrinsics = frame.intrinsics;
    state.camera_to_world = frame.camera_to_world;
    state.gaussians = splat::construct_gaussians(
        frame.rgb, depth_u8, mask_u8, frame.intrinsics, splat::ConstructionConfig{}
    );
    state.validate();

    frame.foreground_mask.pixels[0] = 255;
    require_throws([&]() { frame.validate(); }, "contract mask must enforce binary {0, 1}");
    frame.foreground_mask.pixels[0] = 1;
    frame.camera_to_world[0] = 2.0F;
    require_throws([&]() { frame.validate(); }, "camera pose must reject non-rigid scale");
}

void test_float_relative_depth_and_weights() {
    auto frame = construction_frame(2, 1, {1.0F, 0.0F});
    frame.depth_validity = splat::ImageF32{2, 1, 1, {1.0F, 0.5F}};
    frame.foreground_weight = splat::ImageF32{2, 1, 1, {1.0F, 0.5F}};
    frame.camera_to_world[3] = 10.0F;

    splat::ConstructionConfig config;
    config.sample_stride = 1;
    const auto state = splat::construct_splat_state(frame, config);

    require(state.frame_id == "0001" && state.source_id == "0001"
                && state.timestamp_ns == 123,
            "SplatState must preserve source provenance");
    require(state.gaussians.size() == 2, "both valid relative-depth pixels must construct");
    require(close(state.gaussians.positions[2], -1.0F),
            "relative inverse depth one must map to near distance");
    require(close(state.gaussians.positions[5], -4.0F),
            "relative inverse depth zero must map to far distance");
    require(state.gaussians.positions[0] > 9.0F,
            "camera-to-world translation must be applied to positions");
    require(close(state.gaussians.reconstruction_weights[0], 1.0F)
                && close(state.gaussians.reconstruction_weights[1], 0.25F),
            "validity and foreground weights must multiply");
    require(state.gaussians.opacity_logits[1] < state.gaussians.opacity_logits[0],
            "lower reconstruction weight must initialize lower opacity");
}

void test_metric_depth_and_invalid_rejection() {
    auto frame = construction_frame(
        2, 1, {2.5F, 0.0F}, splat::DepthSemantics::MetricCameraZ
    );
    splat::ConstructionConfig config;
    config.sample_stride = 1;
    const auto state = splat::construct_splat_state(frame, config);

    require(state.gaussians.size() == 1, "zero metric depth must be rejected");
    require(close(state.gaussians.positions[2], -2.5F),
            "metric camera-Z depth must be used without near/far remapping");
    require(state.source_depth_units == splat::DepthUnits::Meters,
            "metric depth units must survive in SplatState");
}

void test_depth_orientation_and_edge_scaling() {
    auto frame = construction_frame(
        3, 3,
        {2.0F, 2.1F, 2.2F, 2.0F, 2.1F, 2.2F, 2.0F, 2.1F, 2.2F},
        splat::DepthSemantics::MetricCameraZ
    );
    splat::ConstructionConfig config;
    config.sample_stride = 1;
    config.depth_edge_threshold = 0.2F;
    config.edge_scale_multiplier = 0.5F;
    const auto state = splat::construct_splat_state(frame, config);

    require(state.gaussians.size() == 9, "sloped metric surface must construct all pixels");
    const std::size_t center_rotation = 4 * 4;
    require(std::abs(state.gaussians.rotations[center_rotation + 2]) > 1.0e-3F,
            "depth gradient must rotate the center Gaussian away from fronto-parallel");
    const float center_scale_x = state.gaussians.log_scales[4 * 3];
    const float edge_scale_x = state.gaussians.log_scales[3 * 3];
    require(center_scale_x > edge_scale_x,
            "missing horizontal support must shrink an edge Gaussian");
}

} // namespace

int main() {
    try {
        test_center_projection_and_attributes();
        test_empty_mask();
        test_reference_compatible_ply_payload();
        test_portable_image_crlf_and_trailing_data();
        test_metadata_is_complete_and_replaceable();
        test_contract_validation();
        test_float_relative_depth_and_weights();
        test_metric_depth_and_invalid_rejection();
        test_depth_orientation_and_edge_scaling();
        std::cout << "All Splat Constructor tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
