# Math module CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Create math library
add_library(tt_math_lib STATIC
    split.cpp
    matmul.cpp
    reduce_sum.cpp
    eltwise.cpp
    transpose.cpp
)

# Set include directories
target_include_directories(tt_math_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/includes
)

# Link with core library
target_link_libraries(tt_math_lib PUBLIC tt_lazy_core)


# Set C++ standard
target_compile_features(tt_math_lib PUBLIC cxx_std_17)

# Create test executable
add_executable(test_math_ops test_math_ops.cpp)
target_link_libraries(test_math_ops tt_math_lib tt_lazy_tape)

# Enable testing
enable_testing()
add_test(NAME math_operations_test COMMAND test_math_ops)
