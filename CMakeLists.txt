cmake_minimum_required(VERSION 3.16)
project(tt_lazy VERSION 1.0.0 LANGUAGES CXX)

# Set default generator to Ninja if not specified
if(NOT CMAKE_GENERATOR)
    set(CMAKE_GENERATOR "Ninja" CACHE STRING "Default generator" FORCE)
endif()

# Enable compile commands export for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags for optimization and debugging
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Core library - basic graph infrastructure
set(CORE_SOURCES
    src/Tensor.cpp
    src/Node.cpp
    src/Context.cpp
    src/MemoryManager.cpp
)

set(CORE_HEADERS
    includes/common.hpp
    includes/Tensor.hpp
    includes/OpArgs.hpp
    includes/Node.hpp
    includes/Context.hpp
    includes/MemoryManager.hpp
    includes/EvaluationManager.hpp
)

# Create core library
add_library(tt_lazy_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})

# Set core library properties
set_target_properties(tt_lazy_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Core library includes
target_include_directories(tt_lazy_core PUBLIC includes)

# Link Boost container to core library
target_link_libraries(tt_lazy_core PUBLIC Boost::container)

# Operations library - depends on core
set(OPERATIONS_SOURCES
    operations/operations.cpp
)

set(OPERATIONS_HEADERS
    includes/operations/operations.hpp
)

# Create operations library
add_library(tt_lazy_operations STATIC ${OPERATIONS_SOURCES} ${OPERATIONS_HEADERS})

# Set operations library properties
set_target_properties(tt_lazy_operations PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Operations library depends on core and inherits its include directories
target_link_libraries(tt_lazy_operations PUBLIC tt_lazy_core)
target_include_directories(tt_lazy_operations PUBLIC includes/operations)

# Math library - immediate computation operations
add_subdirectory(math)

# Tape library - depends on core, operations, and math
add_subdirectory(tape)

# Lazy target - combines core + operations + tape
add_library(tt_lazy_lib INTERFACE)
target_link_libraries(tt_lazy_lib INTERFACE tt_lazy_core tt_lazy_operations tt_lazy_tape)

# Add compiler-specific flags to libraries
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tt_lazy_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(tt_lazy_operations PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Conan integration
find_package(GTest REQUIRED)
find_package(Boost REQUIRED)
find_package(pybind11 REQUIRED)

# Create test executable
add_executable(tt_lazy_tests
    tests/cpp/test_main.cpp
    tests/cpp/test_tensor.cpp
    tests/cpp/test_node.cpp
    tests/cpp/test_context.cpp
    tests/cpp/test_operations.cpp
    tests/cpp/test_end_to_end.cpp
    tests/cpp/test_mlp_demo.cpp
)

# Add include directories for test executable
target_include_directories(tt_lazy_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/includes/operations
    ${CMAKE_SOURCE_DIR}/math
)

# Link test executable with all libraries and gtest
# Note: only link tt_lazy_tape as it transitively provides core and operations
target_link_libraries(tt_lazy_tests
    tt_lazy_tape
    GTest::gtest_main
)

# Set test target properties
set_target_properties(tt_lazy_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add compiler-specific flags to tests
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tt_lazy_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Enable testing
enable_testing()

# Add test to CTest
add_test(NAME tt_lazy_tests COMMAND tt_lazy_tests)

# Python module
pybind11_add_module(tt_lazy_python 
    bindings/python_bindings.cpp
    bindings/core_types.cpp
    bindings/operations.cpp
)
# Only link tt_lazy_tape as it transitively provides core and operations
target_link_libraries(tt_lazy_python PRIVATE tt_lazy_tape pybind11::module)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Optional: Add install targets
install(TARGETS tt_lazy_core tt_lazy_operations tt_lazy_tape DESTINATION lib)
install(DIRECTORY includes/ DESTINATION include/tt_lazy)
install(DIRECTORY tape/ DESTINATION include/tt_lazy/tape FILES_MATCHING PATTERN "*.hpp")
