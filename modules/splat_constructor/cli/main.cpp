#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "splat_constructor/constructor.hpp"
#include "splat_constructor/image_io.hpp"
#include "splat_constructor/ply_writer.hpp"

namespace {

using Arguments = std::unordered_map<std::string, std::string>;

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; index += 2) {
        const std::string key = argv[index];
        if (!key.starts_with("--") || index + 1 >= argc) {
            throw std::invalid_argument("arguments must use --name value pairs");
        }
        arguments[key] = argv[index + 1];
    }
    return arguments;
}

const std::string& require(const Arguments& arguments, const std::string& key) {
    const auto found = arguments.find(key);
    if (found == arguments.end()) {
        throw std::invalid_argument("missing required argument " + key);
    }
    return found->second;
}

std::string optional(
    const Arguments& arguments, const std::string& key, const std::string& fallback
) {
    const auto found = arguments.find(key);
    return found == arguments.end() ? fallback : found->second;
}

std::size_t parse_size(const std::string& text, const char* name) {
    try {
        const auto value = std::stoull(text);
        if (value == 0) {
            throw std::out_of_range("zero");
        }
        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
}

float parse_float(const std::string& text, const char* name) {
    try {
        std::size_t consumed{};
        const float value = std::stof(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be numeric");
    }
}

void print_usage() {
    std::cerr
        << "Usage: splat_constructor_cli --rgb input.ppm --depth depth.pgm "
           "--mask mask.pgm --output splats.ply [options]\n"
        << "Options: --stride 2 --fov 50 --near 1 --far 4 --opacity 0.9 "
           "--scale 0.75 --thickness 0.25 --mask-threshold 128\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const std::filesystem::path rgb_path = require(arguments, "--rgb");
        const std::filesystem::path depth_path = require(arguments, "--depth");
        const std::filesystem::path mask_path = require(arguments, "--mask");
        const std::filesystem::path output_path = require(arguments, "--output");

        splat::ConstructionConfig config;
        config.sample_stride = parse_size(optional(arguments, "--stride", "2"), "stride");
        config.near_depth = parse_float(optional(arguments, "--near", "1"), "near");
        config.far_depth = parse_float(optional(arguments, "--far", "4"), "far");
        config.opacity = parse_float(optional(arguments, "--opacity", "0.9"), "opacity");
        config.scale_multiplier = parse_float(optional(arguments, "--scale", "0.75"), "scale");
        config.thickness_multiplier = parse_float(
            optional(arguments, "--thickness", "0.25"), "thickness"
        );
        const std::size_t threshold = parse_size(
            optional(arguments, "--mask-threshold", "128"), "mask threshold"
        );
        if (threshold > 255) {
            throw std::invalid_argument("mask threshold must not exceed 255");
        }
        config.mask_threshold = static_cast<std::uint8_t>(threshold);

        const splat::ImageU8 rgb = splat::read_portable_image(rgb_path);
        const splat::ImageU8 depth = splat::read_portable_image(depth_path);
        const splat::ImageU8 mask = splat::read_portable_image(mask_path);
        const float fov = parse_float(optional(arguments, "--fov", "50"), "field of view");
        const splat::CameraIntrinsics intrinsics =
            splat::intrinsics_from_horizontal_fov(rgb.width, rgb.height, fov);

        const splat::GaussianSet gaussians =
            splat::construct_gaussians(rgb, depth, mask, intrinsics, config);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        splat::write_3dgs_ply(output_path, gaussians);
        std::filesystem::path metadata_path = output_path;
        metadata_path.replace_extension(".json");
        splat::write_construction_metadata(
            metadata_path, gaussians, intrinsics, config, rgb.width, rgb.height
        );

        std::cout << "Constructed " << gaussians.size() << " Gaussians from "
                  << rgb.width << 'x' << rgb.height << " input.\n"
                  << "PLY: " << output_path.string() << "\n"
                  << "Metadata: " << metadata_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        print_usage();
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
