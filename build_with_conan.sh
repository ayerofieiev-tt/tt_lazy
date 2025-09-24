#!/bin/bash

# Build script using Conan for dependency management

set -e

# Default to dev build if no argument provided
BUILD_TYPE=${1:-dev}

if [ "$BUILD_TYPE" = "dev" ]; then
    echo "Installing dependencies with Conan (dev build)..."
    conan install . --build=missing -s build_type=Debug

    echo "Configuring CMake dev build with Ninja, sanitizers (ASAN+UBSAN), and clang-tidy..."
    cmake --preset dev

    echo "Building dev version with Ninja..."
    cmake --build --preset dev

    echo "Running dev tests with sanitizers and static analysis..."
    ctest --preset dev

    echo "Development build complete with all safety checks!"
elif [ "$BUILD_TYPE" = "release" ]; then
    echo "Installing dependencies with Conan (release build)..."
    conan install . --build=missing -s build_type=Release

    echo "Configuring CMake release build with Ninja (optimized)..."
    cmake --preset release

    echo "Building release version with Ninja..."
    cmake --build --preset release

    echo "Running release tests..."
    ctest --preset release

    echo "Release build complete (optimized)!"
else
    echo "Usage: $0 [dev|release]"
    echo "  dev     - Development build with sanitizers and static analysis (default)"
    echo "  release - Optimized release build"
    exit 1
fi

echo ""
echo "Available build configurations:"
echo "  ./build_with_conan.sh dev     # Development build with all checks"
echo "  ./build_with_conan.sh release # Optimized release build"
