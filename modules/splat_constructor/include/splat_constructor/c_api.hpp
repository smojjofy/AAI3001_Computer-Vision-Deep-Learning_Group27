#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define SPLAT_C_API __declspec(dllexport)
#else
#define SPLAT_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum SplatDepthSemantics {
    SPLAT_RELATIVE_INVERSE = 0,
    SPLAT_METRIC_CAMERA_Z = 1,
    SPLAT_PSEUDO_INVERSE_FROM_SEGMENTATION = 2,
};

enum SplatDepthUnits {
    SPLAT_UNITLESS = 0,
    SPLAT_METERS = 1,
};

typedef struct SplatConstructionConfig {
    size_t sample_stride;
    float near_depth;
    float far_depth;
    uint8_t mask_threshold;
    float opacity;
    float scale_multiplier;
    float thickness_multiplier;
    float depth_edge_threshold;
    float edge_scale_multiplier;
    float minimum_reconstruction_weight;
} SplatConstructionConfig;

typedef struct SplatConstructionInput {
    const char* frame_id;
    uint64_t timestamp_ns;
    const char* source_id;
    size_t width;
    size_t height;
    const uint8_t* rgb;                 // H x W x 3, RGB
    const float* depth;                 // H x W
    const uint8_t* foreground_mask;     // H x W, binary {0, 1}
    const float* depth_validity;        // optional H x W, nullable
    const float* foreground_weight;     // optional H x W, nullable
    float fx;
    float fy;
    float cx;
    float cy;
    int depth_semantics;
    int depth_units;
    const float* camera_to_world;       // 16 row-major float values
} SplatConstructionInput;

typedef struct SplatStateHandle SplatStateHandle;

// Returns null on failure. Read splat_last_error() on the calling thread.
SPLAT_C_API SplatStateHandle* splat_construct(
    const SplatConstructionInput* input,
    const SplatConstructionConfig* config
);
SPLAT_C_API void splat_state_destroy(SplatStateHandle* handle);
SPLAT_C_API const char* splat_last_error(void);

SPLAT_C_API size_t splat_state_size(const SplatStateHandle* handle);
SPLAT_C_API const float* splat_state_positions(const SplatStateHandle* handle);
SPLAT_C_API const float* splat_state_sh_dc(const SplatStateHandle* handle);
SPLAT_C_API const float* splat_state_opacity_logits(const SplatStateHandle* handle);
SPLAT_C_API const float* splat_state_log_scales(const SplatStateHandle* handle);
SPLAT_C_API const float* splat_state_rotations(const SplatStateHandle* handle);
SPLAT_C_API const float* splat_state_reconstruction_weights(const SplatStateHandle* handle);
SPLAT_C_API const uint64_t* splat_state_local_ids(const SplatStateHandle* handle);
SPLAT_C_API const uint32_t* splat_state_source_pixels(const SplatStateHandle* handle);

#ifdef __cplusplus
}
#endif

