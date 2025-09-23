#!/bin/bash

# Build script using Conan for dependency management

set -e

echo "Installing dependencies with Conan..."
conan install . --build=missing

echo "Configuring CMake with Ninja (using preset)..."
cmake --preset conan-release

echo "Building with Ninja..."
cmake --build --preset conan-release

echo "Running tests..."
ctest --preset conan-release

echo "Build complete!"
