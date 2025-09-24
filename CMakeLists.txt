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

# Enable clang-tidy if available
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};--warnings-as-errors=*")
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE} (warnings as errors)")
else()
    message(STATUS "clang-tidy not found - static analysis disabled")
endif()

# Option to disable clang-tidy (useful for faster builds)
option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" ON)
if(NOT ENABLE_CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY "")
endif()

# Sanitizer options - defaults for development builds
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_MSAN "Enable MemorySanitizer" OFF)
option(ENABLE_ALL_SANITIZERS "Enable all compatible sanitizers" OFF)

# Function to add sanitizer flags
function(add_sanitizer_flags target)
    set(SANITIZER_FLAGS "")

    # Note: TSAN and MSAN are mutually exclusive with ASAN
    if(ENABLE_ALL_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # For Clang, we can combine ASAN and UBSAN
            set(SANITIZER_FLAGS "-fsanitize=address,undefined")
        else()
            # For GCC, we also combine ASAN and UBSAN
            set(SANITIZER_FLAGS "-fsanitize=address,undefined")
        endif()
    else()
        if(ENABLE_ASAN AND NOT ENABLE_TSAN AND NOT ENABLE_MSAN)
            set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize=address")
        endif()

        if(ENABLE_UBSAN)
            if(SANITIZER_FLAGS)
                set(SANITIZER_FLAGS "${SANITIZER_FLAGS},undefined")
            else()
                set(SANITIZER_FLAGS "-fsanitize=undefined")
            endif()
        endif()

        if(ENABLE_TSAN AND NOT ENABLE_ASAN AND NOT ENABLE_MSAN)
            set(SANITIZER_FLAGS "-fsanitize=thread")
        endif()

        if(ENABLE_MSAN AND NOT ENABLE_ASAN AND NOT ENABLE_TSAN)
            set(SANITIZER_FLAGS "-fsanitize=memory")
        endif()
    endif()

    if(SANITIZER_FLAGS)
        # Add common sanitizer options
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fno-omit-frame-pointer -g")

        # Add sanitizer-specific options
        if(ENABLE_ASAN OR ENABLE_ALL_SANITIZERS)
            set(SANITIZER_FLAGS "${SANITIZER_FLAGS} -fsanitize-address-use-after-scope")
        endif()

        # Apply flags to target - convert string to list
        separate_arguments(SANITIZER_FLAGS_LIST UNIX_COMMAND "${SANITIZER_FLAGS}")
        target_compile_options(${target} PRIVATE ${SANITIZER_FLAGS_LIST})
        target_link_options(${target} PRIVATE ${SANITIZER_FLAGS_LIST})
    endif()
endfunction()

# Enhanced compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(ENHANCED_WARNINGS
        -Wall -Wextra -Wpedantic -Werror
        -Wcast-align -Wcast-qual -Wctor-dtor-privacy
        -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op
        -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls
        -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel
        -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(ENHANCED_WARNINGS
        -Wall -Wextra -Wpedantic -Werror
        -Wcast-align -Wcast-qual -Wctor-dtor-privacy
        -Wdisabled-optimization -Wformat=2 -Winit-self
        -Wmissing-declarations -Wmissing-include-dirs
        -Wold-style-cast -Woverloaded-virtual -Wredundant-decls
        -Wshadow -Wsign-conversion -Wsign-promo
        -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused
        -Wnull-dereference -Wdouble-promotion
    )
endif()

# Set compiler flags for optimization and debugging
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    # Convert the warning list to a string and append
    string(JOIN " " WARNING_FLAGS ${ENHANCED_WARNINGS})
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${WARNING_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Werror")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -Werror")
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set default sanitizer behavior based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT ENABLE_ASAN AND NOT ENABLE_UBSAN AND NOT ENABLE_TSAN AND NOT ENABLE_MSAN)
    # For debug builds, enable ASAN+UBSAN by default if no sanitizers are explicitly set
    set(ENABLE_ASAN ON)
    set(ENABLE_UBSAN ON)
    message(STATUS "Debug build detected - enabling ASAN+UBSAN by default")
endif()

# Core library - basic graph infrastructure
set(CORE_SOURCES
    src/core/Tensor.cpp
    src/core/Node.cpp
    src/core/Context.cpp
    src/core/MemoryManager.cpp
)

set(CORE_HEADERS
    src/core/common.hpp
    src/core/Tensor.hpp
    src/core/OpArgs.hpp
    src/core/Node.hpp
    src/core/Context.hpp
    src/core/MemoryManager.hpp
    src/core/EvaluationManager.hpp
)

# Create core library
add_library(tt_lazy_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})

# Set core library properties
set_target_properties(tt_lazy_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Core library includes
target_include_directories(tt_lazy_core PUBLIC src/core)

# Link Boost container and spdlog to core library
target_link_libraries(tt_lazy_core PUBLIC Boost::container spdlog::spdlog)

# Apply sanitizers to core library
add_sanitizer_flags(tt_lazy_core)

# Operations library - depends on core
set(OPERATIONS_SOURCES
    src/frontend/operations.cpp
)

set(OPERATIONS_HEADERS
    src/frontend/operations.hpp
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
target_include_directories(tt_lazy_operations PUBLIC src/frontend)

# Apply sanitizers to operations library
add_sanitizer_flags(tt_lazy_operations)

# Math library - immediate computation operations
set(MATH_SOURCES
    src/backend/cpu/split.cpp
    src/backend/cpu/matmul.cpp
    src/backend/cpu/reduce_sum.cpp
    src/backend/cpu/eltwise.cpp
    src/backend/cpu/transpose.cpp
    src/backend/cpu/fused_ops.cpp
)

# Create math library
add_library(tt_math_lib STATIC ${MATH_SOURCES})

# Set math library properties
set_target_properties(tt_math_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Math library includes and dependencies
target_include_directories(tt_math_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src/backend/cpu
)
target_link_libraries(tt_math_lib PUBLIC tt_lazy_core)

# Apply sanitizers to math library
add_sanitizer_flags(tt_math_lib)

# Add compiler-specific flags to math library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tt_math_lib PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Tape library - depends on core, operations, and math
set(TAPE_SOURCES
    src/tape/Tape.cpp
    src/tape/TapeGenerator.cpp
    src/tape/TapeExecutor.cpp
    src/tape/TapeEvaluationManager.cpp
    src/tape/OperationHandlers.cpp
    src/tape/passes/TapeOptimizationPass.cpp
    src/tape/passes/DeadCodeEliminationPass.cpp
    src/tape/passes/MLPFusionPass.cpp
)

# Create tape library
add_library(tt_lazy_tape STATIC ${TAPE_SOURCES})

# Set tape library properties
set_target_properties(tt_lazy_tape PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Tape library includes and dependencies
target_include_directories(tt_lazy_tape PUBLIC
    ${CMAKE_SOURCE_DIR}/src/tape
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/frontend
    ${CMAKE_SOURCE_DIR}/src/backend/cpu
)

# Tape library depends on core, operations, and math libraries
# Math library is PRIVATE as it's an implementation detail
target_link_libraries(tt_lazy_tape
    PUBLIC
        tt_lazy_core
        tt_lazy_operations
    PRIVATE
        tt_math_lib
)

# Apply sanitizers to tape library
add_sanitizer_flags(tt_lazy_tape)

# Add compiler-specific flags to tape library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tt_lazy_tape PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Lazy target - combines core + operations + tape
add_library(tt_lazy_lib INTERFACE)
target_link_libraries(tt_lazy_lib INTERFACE tt_lazy_core tt_lazy_operations tt_lazy_tape)

# Add compiler-specific flags to libraries
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tt_lazy_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(tt_lazy_operations PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Conan integration
find_package(GTest REQUIRED)
find_package(Boost REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)

# Create test executable
add_executable(tt_lazy_tests
    tests/cpp/test_main.cpp
    tests/cpp/unit/test_tensor.cpp
    tests/cpp/unit/test_node.cpp
    tests/cpp/unit/test_context.cpp
    tests/cpp/unit/math/test_math_ops.cpp
    tests/cpp/integration/test_operations.cpp
    tests/cpp/integration/test_end_to_end.cpp
    tests/cpp/benchmarks/test_mlp_demo.cpp
)

# Add include directories for test executable
target_include_directories(tt_lazy_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/frontend
    ${CMAKE_SOURCE_DIR}/src/backend/cpu
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/tape
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

# Apply sanitizers to test executable
add_sanitizer_flags(tt_lazy_tests)

# Add compiler-specific flags to tests
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tt_lazy_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Enable testing
enable_testing()

# Add test to CTest
add_test(NAME tt_lazy_tests COMMAND tt_lazy_tests)

# Python module
pybind11_add_module(tt_lazy_python
    src/bindings/python_bindings.cpp
    src/bindings/core_types.cpp
    src/bindings/operations.cpp
)
# Only link tt_lazy_tape as it transitively provides core and operations
target_link_libraries(tt_lazy_python PRIVATE tt_lazy_tape pybind11::module)

# Apply sanitizers to Python module
add_sanitizer_flags(tt_lazy_python)

# Add compiler-specific flags to Python module
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tt_lazy_python PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Print sanitizer status
message(STATUS "Sanitizers enabled:")
if(ENABLE_ALL_SANITIZERS)
    message(STATUS "  All compatible sanitizers: ON (ASAN + UBSAN)")
else()
    message(STATUS "  AddressSanitizer (ASAN): ${ENABLE_ASAN}")
    message(STATUS "  UndefinedBehaviorSanitizer (UBSAN): ${ENABLE_UBSAN}")
    message(STATUS "  ThreadSanitizer (TSAN): ${ENABLE_TSAN}")
    message(STATUS "  MemorySanitizer (MSAN): ${ENABLE_MSAN}")
endif()
message(STATUS "Static analysis:")
message(STATUS "  Clang-Tidy: ${ENABLE_CLANG_TIDY}")

# Optional: Add install targets
install(TARGETS tt_lazy_core tt_lazy_operations tt_lazy_tape DESTINATION lib)
install(DIRECTORY src/core/ DESTINATION include/tt_lazy/core FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY src/frontend/ DESTINATION include/tt_lazy/frontend FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY src/tape/ DESTINATION include/tt_lazy/tape FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY src/backend/cpu/ DESTINATION include/tt_lazy/backend/cpu FILES_MATCHING PATTERN "*.hpp")

# Custom target to run clang-tidy manually on all source files
if(CLANG_TIDY_EXE)
    # Collect all source files
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/core/*.cpp
        ${CMAKE_SOURCE_DIR}/src/core/*.hpp
        ${CMAKE_SOURCE_DIR}/src/frontend/*.cpp
        ${CMAKE_SOURCE_DIR}/src/frontend/*.hpp
        ${CMAKE_SOURCE_DIR}/src/tape/*.cpp
        ${CMAKE_SOURCE_DIR}/src/tape/*.hpp
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/*.cpp
        ${CMAKE_SOURCE_DIR}/src/backend/cpu/*.hpp
        ${CMAKE_SOURCE_DIR}/src/bindings/*.cpp
    )

    # Create custom target for running clang-tidy
    add_custom_target(
        clang-tidy
        COMMAND ${CLANG_TIDY_EXE}
        -p ${CMAKE_BINARY_DIR}
        ${ALL_SOURCE_FILES}
        COMMENT "Running clang-tidy on all source files"
        VERBATIM
    )
endif()
