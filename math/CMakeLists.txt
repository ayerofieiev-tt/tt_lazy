# Math module CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Boost will be provided by Conan via target linking

# Create math library
add_library(tt_math_lib STATIC
    split.cpp
    matmul.cpp
    reduce_sum.cpp
    eltwise.cpp
    transpose.cpp
    fused_ops.cpp
)

# Set include directories
target_include_directories(tt_math_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/includes/math
)

# Link with core library
target_link_libraries(tt_math_lib PUBLIC tt_lazy_core)


# Set C++ standard
target_compile_features(tt_math_lib PUBLIC cxx_std_17)