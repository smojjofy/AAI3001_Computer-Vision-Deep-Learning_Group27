#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
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

struct ConstructionConfig {
    std::size_t sample_stride{2};
    float near_depth{1.0F};
    float far_depth{4.0F};
    std::uint8_t mask_threshold{128};
    float opacity{0.9F};
    float scale_multiplier{0.75F};
    float thickness_multiplier{0.25F};
    DepthSemantics depth_semantics{DepthSemantics::PseudoInverseFromSegmentation};
};

// Structure-of-arrays storage keeps attributes contiguous for future SIMD/GPU use.
struct GaussianSet {
    std::vector<float> positions;      // xyz
    std::vector<float> sh_dc;          // RGB degree-0 SH coefficients
    std::vector<float> opacity_logits;
    std::vector<float> log_scales;     // xyz
    std::vector<float> rotations;      // quaternion wxyz
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
        source_pixels.reserve(count * 2);
    }

    void validate() const;
};

} // namespace splat
