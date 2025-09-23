# Tape library CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Create tape library
add_library(tt_lazy_tape STATIC
    Tape.cpp
    TapeGenerator.cpp
    TapeExecutor.cpp
    TapeEvaluationManager.cpp
    OperationHandlers.cpp
)

# Set include directories
target_include_directories(tt_lazy_tape PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/includes
    ${CMAKE_SOURCE_DIR}/includes/tape
)

# Set C++ standard
target_compile_features(tt_lazy_tape PUBLIC cxx_std_17)

# Tape library depends on core, operations, and math libraries
target_link_libraries(tt_lazy_tape PRIVATE 
    tt_lazy_core
    tt_lazy_operations 
    tt_math_lib
)

# Add compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tt_lazy_tape PRIVATE -Wall -Wextra -Wpedantic)
endif()
