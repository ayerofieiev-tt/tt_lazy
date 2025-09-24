#!/bin/bash

# Script to run clang-tidy on the entire codebase
# Usage: ./run-clang-tidy.sh [options]
# Options:
#   --fix    Apply fixes automatically
#   --build-dir <dir>  Specify build directory (default: build)

set -e

# Default values
BUILD_DIR="build"
FIX_FLAG=""
PARALLEL_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_FLAG="--fix"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --jobs|-j)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fix] [--build-dir <dir>] [--jobs <num>]"
            exit 1
            ;;
    esac
done

# Check if clang-tidy is installed
CLANG_TIDY_PATH=""
if command -v clang-tidy &> /dev/null; then
    CLANG_TIDY_PATH="clang-tidy"
elif [ -f "/opt/homebrew/opt/llvm/bin/clang-tidy" ]; then
    CLANG_TIDY_PATH="/opt/homebrew/opt/llvm/bin/clang-tidy"
else
    echo "Error: clang-tidy is not installed or not in PATH"
    echo "Please install clang-tidy to use this script"
    exit 1
fi

# Check if build directory exists and has compile_commands.json
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' does not exist"
    echo "Please run cmake to configure the project first"
    exit 1
fi

if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
    echo "Error: compile_commands.json not found in '$BUILD_DIR'"
    echo "Please ensure CMAKE_EXPORT_COMPILE_COMMANDS=ON is set"
    exit 1
fi

echo "Running clang-tidy with the following configuration:"
echo "  Build directory: $BUILD_DIR"
echo "  Fix mode: $([ -n "$FIX_FLAG" ] && echo "enabled" || echo "disabled")"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Clang-tidy version: $($CLANG_TIDY_PATH --version | head -n 1)"
echo ""

# Find all source files
SOURCE_FILES=$(find src includes operations tape math bindings -name "*.cpp" -o -name "*.hpp" -o -name "*.h" 2>/dev/null | grep -v "build" | sort)

# Count files
FILE_COUNT=$(echo "$SOURCE_FILES" | wc -l)
echo "Found $FILE_COUNT source files to analyze"
echo ""

# Run clang-tidy
if [ -n "$FIX_FLAG" ]; then
    echo "Running clang-tidy with automatic fixes..."
    echo "$SOURCE_FILES" | xargs -P "$PARALLEL_JOBS" -I {} $CLANG_TIDY_PATH -p "$BUILD_DIR" $FIX_FLAG {}
else
    echo "Running clang-tidy in check mode..."
    echo "$SOURCE_FILES" | xargs -P "$PARALLEL_JOBS" -I {} $CLANG_TIDY_PATH -p "$BUILD_DIR" {}
fi

echo ""
echo "Clang-tidy analysis complete!"
