#include "math_operations.hpp"

#include <iostream>
#include <vector>

#include <spdlog/spdlog.h>

int main() {
    using namespace math;

    spdlog::info("=== Testing Math-Based Operations ===");

    // Test 1: Create tensors with data
    spdlog::info("\n1. Creating tensors with data:");

    // Create a 1D tensor
    Tensor a({4});
    float* a_data = a.data_ptr();
    for (size_t i = 0; i < 4; ++i) {
        a_data[i] = static_cast<float>(i + 1);
    }
    spdlog::info("Tensor a: ");
    a.print();

    // Create a 2D tensor
    Tensor b({2, 3});
    float* b_data = b.data_ptr();
    for (size_t i = 0; i < 6; ++i) {
        b_data[i] = static_cast<float>(i + 1);
    }
    spdlog::info("Tensor b: ");
    b.print();

    // Test 2: ReLU operation
    spdlog::info("\n2. Testing ReLU operation:");
    Tensor c({4});
    float* c_data = c.data_ptr();
    c_data[0] = -2.0f;
    c_data[1] = -1.0f;
    c_data[2] = 0.0f;
    c_data[3] = 3.0f;

    spdlog::info("Input to ReLU: ");
    c.print();

    Tensor relu_result = relu(c);
    spdlog::info("ReLU output: ");
    relu_result.print();

    // Test 3: Reduce sum
    spdlog::info("\n3. Testing reduce sum:");
    Tensor d({3});
    float* d_data = d.data_ptr();
    d_data[0] = 1.0f;
    d_data[1] = 2.0f;
    d_data[2] = 3.0f;

    spdlog::info("Input to reduce_sum: ");
    d.print();

    Tensor sum_result = reduce_sum(d, {0});
    spdlog::info("Sum result: ");
    sum_result.print();

    // Test 4: Matrix multiplication
    spdlog::info("\n4. Testing matrix multiplication:");
    Tensor e({2, 3});
    Tensor f({3, 2});

    // Fill matrices
    float* e_data = e.data_ptr();
    float* f_data = f.data_ptr();
    for (size_t i = 0; i < 6; ++i) {
        e_data[i] = static_cast<float>(i + 1);
        f_data[i] = static_cast<float>(i + 1);
    }

    spdlog::info("Matrix e (2x3): ");
    e.print();
    spdlog::info("Matrix f (3x2): ");
    f.print();

    Tensor matmul_result = matmul(e, f);
    spdlog::info("Matrix multiplication result (2x2): ");
    matmul_result.print();

    // Test 5: Element-wise operations
    spdlog::info("\n5. Testing element-wise operations:");
    Tensor g({3});
    Tensor h({3});

    float* g_data = g.data_ptr();
    float* h_data = h.data_ptr();
    for (size_t i = 0; i < 3; ++i) {
        g_data[i] = static_cast<float>(i + 1);
        h_data[i] = static_cast<float>(i + 2);
    }

    spdlog::info("Tensor g: ");
    g.print();
    spdlog::info("Tensor h: ");
    h.print();

    Tensor add_result = add(g, h);
    spdlog::info("Addition result: ");
    add_result.print();

    Tensor mul_result = multiply(g, h);
    spdlog::info("Multiplication result: ");
    mul_result.print();

    spdlog::info("\n=== All tests completed ===");

    return 0;
}
