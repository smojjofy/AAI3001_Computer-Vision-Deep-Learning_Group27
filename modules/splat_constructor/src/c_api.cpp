#include "splat_constructor/c_api.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "splat_constructor/constructor.hpp"

struct SplatStateHandle {
    splat::SplatState state;
};

namespace {

thread_local std::string last_error;

std::size_t checked_pixel_count(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0 || width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::invalid_argument("width and height must be non-zero without overflow");
    }
    return width * height;
}

template <typename T>
std::vector<T> copy_plane(const T* values, std::size_t count, const char* name) {
    if (values == nullptr) {
        throw std::invalid_argument(std::string(name) + " must not be null");
    }
    return std::vector<T>(values, values + count);
}

splat::DepthSemantics depth_semantics(int value) {
    switch (value) {
    case SPLAT_RELATIVE_INVERSE:
        return splat::DepthSemantics::RelativeInverse;
    case SPLAT_METRIC_CAMERA_Z:
        return splat::DepthSemantics::MetricCameraZ;
    case SPLAT_PSEUDO_INVERSE_FROM_SEGMENTATION:
        return splat::DepthSemantics::PseudoInverseFromSegmentation;
    default:
        throw std::invalid_argument("unknown depth semantics");
    }
}

splat::DepthUnits depth_units(int value) {
    switch (value) {
    case SPLAT_UNITLESS:
        return splat::DepthUnits::Unitless;
    case SPLAT_METERS:
        return splat::DepthUnits::Meters;
    default:
        throw std::invalid_argument("unknown depth units");
    }
}

splat::ConstructionConfig construction_config(const SplatConstructionConfig& input) {
    splat::ConstructionConfig output;
    output.sample_stride = input.sample_stride;
    output.near_depth = input.near_depth;
    output.far_depth = input.far_depth;
    output.mask_threshold = input.mask_threshold;
    output.opacity = input.opacity;
    output.scale_multiplier = input.scale_multiplier;
    output.thickness_multiplier = input.thickness_multiplier;
    output.depth_edge_threshold = input.depth_edge_threshold;
    output.edge_scale_multiplier = input.edge_scale_multiplier;
    output.minimum_reconstruction_weight = input.minimum_reconstruction_weight;
    return output;
}

splat::ConstructionFrame construction_frame(const SplatConstructionInput& input) {
    if (input.frame_id == nullptr || input.source_id == nullptr || input.camera_to_world == nullptr) {
        throw std::invalid_argument("frame ID, source ID, and camera pose must not be null");
    }
    const std::size_t pixels = checked_pixel_count(input.width, input.height);
    if (pixels > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::invalid_argument("RGB image dimensions overflow memory size");
    }

    splat::ConstructionFrame frame;
    frame.frame_id = input.frame_id;
    frame.timestamp_ns = input.timestamp_ns;
    frame.source_id = input.source_id;
    frame.rgb = splat::ImageU8{
        input.width, input.height, 3, copy_plane(input.rgb, pixels * 3, "rgb")
    };
    frame.depth = splat::ImageF32{
        input.width, input.height, 1, copy_plane(input.depth, pixels, "depth")
    };
    frame.foreground_mask = splat::ImageU8{
        input.width, input.height, 1, copy_plane(input.foreground_mask, pixels, "foreground mask")
    };
    if (input.depth_validity != nullptr) {
        frame.depth_validity = splat::ImageF32{
            input.width, input.height, 1, copy_plane(input.depth_validity, pixels, "depth validity")
        };
    }
    if (input.foreground_weight != nullptr) {
        frame.foreground_weight = splat::ImageF32{
            input.width, input.height, 1,
            copy_plane(input.foreground_weight, pixels, "foreground weight")
        };
    }
    frame.intrinsics = splat::CameraIntrinsics{input.fx, input.fy, input.cx, input.cy};
    frame.depth_semantics = depth_semantics(input.depth_semantics);
    frame.depth_units = depth_units(input.depth_units);
    std::copy_n(input.camera_to_world, frame.camera_to_world.size(), frame.camera_to_world.begin());
    return frame;
}

} // namespace

extern "C" SplatStateHandle* splat_construct(
    const SplatConstructionInput* input,
    const SplatConstructionConfig* config
) {
    last_error.clear();
    try {
        if (input == nullptr || config == nullptr) {
            throw std::invalid_argument("input and configuration must not be null");
        }
        auto handle = std::make_unique<SplatStateHandle>();
        handle->state = splat::construct_splat_state(
            construction_frame(*input), construction_config(*config)
        );
        return handle.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown native Splat Constructor failure";
        return nullptr;
    }
}

extern "C" void splat_state_destroy(SplatStateHandle* handle) {
    delete handle;
}

extern "C" const char* splat_last_error(void) {
    return last_error.c_str();
}

extern "C" size_t splat_state_size(const SplatStateHandle* handle) {
    return handle == nullptr ? 0 : handle->state.gaussians.size();
}

extern "C" const float* splat_state_positions(const SplatStateHandle* handle) {
    return handle == nullptr ? nullptr : handle->state.gaussians.positions.data();
}

extern "C" const float* splat_state_sh_dc(const SplatStateHandle* handle) {
    return handle == nullptr ? nullptr : handle->state.gaussians.sh_dc.data();
}

extern "C" const float* splat_state_opacity_logits(const SplatStateHandle* handle) {
    return handle == nullptr ? nullptr : handle->state.gaussians.opacity_logits.data();
}

extern "C" const float* splat_state_log_scales(const SplatStateHandle* handle) {
    return handle == nullptr ? nullptr : handle->state.gaussians.log_scales.data();
}

extern "C" const float* splat_state_rotations(const SplatStateHandle* handle) {
    return handle == nullptr ? nullptr : handle->state.gaussians.rotations.data();
}

extern "C" const float* splat_state_reconstruction_weights(const SplatStateHandle* handle) {
    return handle == nullptr ? nullptr : handle->state.gaussians.reconstruction_weights.data();
}

extern "C" const uint64_t* splat_state_local_ids(const SplatStateHandle* handle) {
    return handle == nullptr ? nullptr : handle->state.gaussians.local_ids.data();
}

extern "C" const uint32_t* splat_state_source_pixels(const SplatStateHandle* handle) {
    return handle == nullptr ? nullptr : handle->state.gaussians.source_pixels.data();
}
