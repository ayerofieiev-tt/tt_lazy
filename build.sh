#!/bin/bash

# Build script using Conan for dependency management

set -e

# Default to dev build if no argument provided
BUILD_TYPE=${1:-dev}

# Validate build type
if [ "$BUILD_TYPE" != "dev" ] && [ "$BUILD_TYPE" != "release" ]; then
    echo "Usage: $0 [dev|release]"
    echo "  dev     - Development build with sanitizers and static analysis (default)"
    echo "  release - Optimized release build"
    exit 1
fi

# Set build configuration based on type
if [ "$BUILD_TYPE" = "dev" ]; then
    CONAN_BUILD_TYPE="Debug"
    CMAKE_PRESET="conan-debug"
    BUILD_DIR="build/Debug"
    BUILD_DESCRIPTION="dev build with Ninja, sanitizers (ASAN+UBSAN), and clang-tidy"
    TEST_DESCRIPTION="dev tests with sanitizers and static analysis"
    COMPLETION_MESSAGE="Development build complete with all safety checks!"
else
    CONAN_BUILD_TYPE="Release"
    CMAKE_PRESET="conan-release"
    BUILD_DIR="build/Release"
    BUILD_DESCRIPTION="release build with Ninja (optimized)"
    TEST_DESCRIPTION="release tests"
    COMPLETION_MESSAGE="Release build complete (optimized)!"
fi

echo "Installing dependencies with Conan ($BUILD_TYPE build)..."
conan install . --build=missing -s build_type=$CONAN_BUILD_TYPE

echo "Configuring CMake $BUILD_DESCRIPTION..."
cmake --preset $CMAKE_PRESET -G Ninja

echo "Building $BUILD_TYPE version with Ninja..."
cmake --build $BUILD_DIR

echo "Running $TEST_DESCRIPTION..."
ctest --test-dir $BUILD_DIR

echo "$COMPLETION_MESSAGE"

echo ""
echo "Available build configurations:"
echo "  ./build.sh dev     # Development build with all checks"
echo "  ./build.sh release # Optimized release build"
