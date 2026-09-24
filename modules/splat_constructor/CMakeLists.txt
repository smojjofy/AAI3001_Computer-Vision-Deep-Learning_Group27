cmake_minimum_required(VERSION 3.20)
project(splat_constructor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(splat_constructor STATIC
    src/constructor.cpp
    src/image_io.cpp
    src/ply_writer.cpp
)
target_include_directories(splat_constructor PUBLIC include)

if(MSVC)
    target_compile_options(splat_constructor PRIVATE /W4 /permissive-)
else()
    target_compile_options(splat_constructor PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(splat_constructor_cli cli/main.cpp)
target_link_libraries(splat_constructor_cli PRIVATE splat_constructor)

include(CTest)
if(BUILD_TESTING)
    add_executable(splat_constructor_tests tests/test_constructor.cpp)
    target_link_libraries(splat_constructor_tests PRIVATE splat_constructor)
    add_test(NAME splat_constructor_tests COMMAND splat_constructor_tests)
endif()
