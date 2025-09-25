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
    src/core/tensor.cpp
    src/core/shape.cpp
    src/core/graph_utils.cpp
    src/core/evaluation_manager.cpp
)

set(CORE_HEADERS
    src/core/common.hpp
    src/core/tensor.hpp
    src/core/op_args.hpp
    src/core/evaluation_manager.hpp
    src/core/shape.hpp
    src/core/graph_utils.hpp
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

# Add example executable
add_executable(graph_visualization_demo examples/graph_visualization_demo.cpp)
target_link_libraries(graph_visualization_demo tt_lazy_operations)
target_include_directories(graph_visualization_demo PRIVATE src)

# Apply sanitizers to executable
add_sanitizer_flags(graph_visualization_demo)

# Lazy target - combines core + operations + tape
add_library(tt_lazy_lib INTERFACE)
target_link_libraries(tt_lazy_lib INTERFACE tt_lazy_core tt_lazy_operations)

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
install(TARGETS tt_lazy_core tt_lazy_operations DESTINATION lib)
install(DIRECTORY src/core/ DESTINATION include/tt_lazy/core FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY src/frontend/ DESTINATION include/tt_lazy/frontend FILES_MATCHING PATTERN "*.hpp")

# Custom target to run clang-tidy manually on all source files
if(CLANG_TIDY_EXE)
    # Collect all source files
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/core/*.cpp
        ${CMAKE_SOURCE_DIR}/src/core/*.hpp
        ${CMAKE_SOURCE_DIR}/src/frontend/*.cpp
        ${CMAKE_SOURCE_DIR}/src/frontend/*.hpp
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
