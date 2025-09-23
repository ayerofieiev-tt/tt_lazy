#!/bin/bash

# Build script using Conan for dependency management

set -e

echo "Installing dependencies with Conan..."
conan install . --build=missing

echo "Configuring CMake..."
cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

echo "Building..."
cmake --build build --config Release

echo "Running tests..."
cd build && ctest --output-on-failure

echo "Build complete!"
