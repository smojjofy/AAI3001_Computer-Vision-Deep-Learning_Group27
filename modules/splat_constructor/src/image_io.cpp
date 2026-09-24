#include "splat_constructor/image_io.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace splat {
namespace {

std::string read_token(std::istream& input) {
    std::string token;
    char character{};
    while (input.get(character)) {
        if (character == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(character))) {
            token.push_back(character);
            break;
        }
    }
    while (input.get(character)) {
        if (std::isspace(static_cast<unsigned char>(character))) {
            break;
        }
        token.push_back(character);
    }
    if (token.empty()) {
        throw std::runtime_error("unexpected end of portable image header");
    }
    return token;
}

std::size_t parse_size(const std::string& token, const char* field) {
    try {
        const auto value = std::stoull(token);
        if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
            throw std::out_of_range("size");
        }
        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid portable image ") + field);
    }
}

} // namespace

ImageU8 read_portable_image(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open image: " + path.string());
    }

    const std::string magic = read_token(input);
    const std::size_t channels = magic == "P6" ? 3 : magic == "P5" ? 1 : 0;
    if (channels == 0) {
        throw std::runtime_error("only binary PPM (P6) and PGM (P5) are supported");
    }
    const std::size_t width = parse_size(read_token(input), "width");
    const std::size_t height = parse_size(read_token(input), "height");
    const std::size_t maximum = parse_size(read_token(input), "maximum value");
    if (maximum != 255) {
        throw std::runtime_error("only 8-bit portable images with max value 255 are supported");
    }
    if (width > std::numeric_limits<std::size_t>::max() / height / channels) {
        throw std::runtime_error("portable image dimensions overflow memory size");
    }

    ImageU8 image{width, height, channels, std::vector<std::uint8_t>(width * height * channels)};
    input.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        throw std::runtime_error("portable image payload is truncated");
    }
    return image;
}

} // namespace splat
