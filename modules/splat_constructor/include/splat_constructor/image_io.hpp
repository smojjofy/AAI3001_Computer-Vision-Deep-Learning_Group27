#pragma once

#include <filesystem>

#include "splat_constructor/types.hpp"

namespace splat {

// Reads binary PPM (P6) and PGM (P5) files with 8-bit samples.
[[nodiscard]] ImageU8 read_portable_image(const std::filesystem::path& path);

} // namespace splat
