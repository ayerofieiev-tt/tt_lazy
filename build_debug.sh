#!/bin/bash

# Debug build script

set -e

echo "Installing dependencies with Conan (debug)..."
conan install . --build=missing -s build_type=Debug

echo "Configuring CMake debug build with Ninja..."
cmake --preset debug

echo "Building debug version with Ninja..."
cmake --build --preset debug

echo "Running debug tests..."
ctest --preset debug

echo "Debug build complete!"
