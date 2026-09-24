#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace splat {

struct ImageU8 {
    std::size_t width{};
    std::size_t height{};
    std::size_t channels{};
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] std::uint8_t at(
        std::size_t x, std::size_t y, std::size_t channel = 0
    ) const {
        if (x >= width || y >= height || channel >= channels) {
            throw std::out_of_range("image coordinate is out of range");
        }
        return pixels[(y * width + x) * channels + channel];
    }
};

struct ImageF32 {
    std::size_t width{};
    std::size_t height{};
    std::size_t channels{};
    std::vector<float> pixels;
};

struct CameraIntrinsics {
    float fx{};
    float fy{};
    float cx{};
    float cy{};
};

enum class DepthSemantics {
    RelativeInverse,
    MetricCameraZ,
    PseudoInverseFromSegmentation,
};

enum class DepthUnits {
    Unitless,
    Meters,
};

struct ConstructionConfig {
    std::size_t sample_stride{2};
    float near_depth{1.0F};
    float far_depth{4.0F};
    std::uint8_t mask_threshold{128};
    float opacity{0.9F};
    float scale_multiplier{0.75F};
    float thickness_multiplier{0.25F};
    float depth_edge_threshold{0.1F};
    float edge_scale_multiplier{0.5F};
    float minimum_reconstruction_weight{0.0F};
    // Used by the legacy 8-bit adapter. ConstructionFrame owns semantics in v1.
    DepthSemantics depth_semantics{DepthSemantics::PseudoInverseFromSegmentation};
};

// Structure-of-arrays storage keeps attributes contiguous for future SIMD/GPU use.
struct GaussianSet {
    std::vector<float> positions;      // xyz
    std::vector<float> sh_dc;          // RGB degree-0 SH coefficients
    std::vector<float> opacity_logits;
    std::vector<float> log_scales;     // xyz
    std::vector<float> rotations;      // quaternion wxyz
    std::vector<float> reconstruction_weights; // [0, 1], not serialized to PLY
    std::vector<std::uint64_t> local_ids; // stable within a frame; source pixel index
    std::vector<std::uint32_t> source_pixels; // uv, not serialized to 3DGS PLY

    [[nodiscard]] std::size_t size() const noexcept {
        return opacity_logits.size();
    }

    void reserve(std::size_t count) {
        positions.reserve(count * 3);
        sh_dc.reserve(count * 3);
        opacity_logits.reserve(count);
        log_scales.reserve(count * 3);
        rotations.reserve(count * 4);
        reconstruction_weights.reserve(count);
        local_ids.reserve(count);
        source_pixels.reserve(count * 2);
    }

    void validate() const;
};

// Version 1 input boundary. Masks are binary (object = 1); optional weight and
// validity images are empty when unavailable and otherwise contain one float per pixel.
struct ConstructionFrame {
    static constexpr std::uint32_t schema_version = 1;

    std::string frame_id;
    std::uint64_t timestamp_ns{};
    std::string source_id;
    ImageU8 rgb;
    ImageF32 depth;
    ImageU8 foreground_mask;
    ImageF32 depth_validity;
    ImageF32 foreground_weight;
    CameraIntrinsics intrinsics;
    DepthSemantics depth_semantics{DepthSemantics::RelativeInverse};
    DepthUnits depth_units{DepthUnits::Unitless};
    std::array<float, 16> camera_to_world{
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    };

    void validate() const;
};

// Version 1 runtime output boundary. PLY and JSON are exports, not this contract.
struct SplatState {
    static constexpr std::uint32_t schema_version = 1;

    std::string frame_id;
    std::uint64_t timestamp_ns{};
    std::string source_id;
    std::size_t source_width{};
    std::size_t source_height{};
    CameraIntrinsics intrinsics;
    std::array<float, 16> camera_to_world{};
    DepthSemantics source_depth_semantics{DepthSemantics::RelativeInverse};
    DepthUnits source_depth_units{DepthUnits::Unitless};
    GaussianSet gaussians;

    void validate() const;
};

} // namespace splat
