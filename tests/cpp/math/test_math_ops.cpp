#include "math_operations.hpp"
#include <iostream>
#include <vector>

int main() {
    using namespace math;
    
    std::cout << "=== Testing Math-Based Operations ===" << std::endl;
    
    // Test 1: Create tensors with data
    std::cout << "\n1. Creating tensors with data:" << std::endl;
    
    // Create a 1D tensor
    Tensor a({4});
    float* a_data = a.data_ptr();
    for (size_t i = 0; i < 4; ++i) {
        a_data[i] = static_cast<float>(i + 1);
    }
    std::cout << "Tensor a: ";
    a.print();
    
    // Create a 2D tensor
    Tensor b({2, 3});
    float* b_data = b.data_ptr();
    for (size_t i = 0; i < 6; ++i) {
        b_data[i] = static_cast<float>(i + 1);
    }
    std::cout << "Tensor b: ";
    b.print();
    
    // Test 2: ReLU operation
    std::cout << "\n2. Testing ReLU operation:" << std::endl;
    Tensor c({4});
    float* c_data = c.data_ptr();
    c_data[0] = -2.0f;
    c_data[1] = -1.0f;
    c_data[2] = 0.0f;
    c_data[3] = 3.0f;
    
    std::cout << "Input to ReLU: ";
    c.print();
    
    Tensor relu_result = relu(c);
    std::cout << "ReLU output: ";
    relu_result.print();
    
    // Test 3: Reduce sum
    std::cout << "\n3. Testing reduce sum:" << std::endl;
    Tensor d({3});
    float* d_data = d.data_ptr();
    d_data[0] = 1.0f;
    d_data[1] = 2.0f;
    d_data[2] = 3.0f;
    
    std::cout << "Input to reduce_sum: ";
    d.print();
    
    Tensor sum_result = reduce_sum(d, {0});
    std::cout << "Sum result: ";
    sum_result.print();
    
    // Test 4: Matrix multiplication
    std::cout << "\n4. Testing matrix multiplication:" << std::endl;
    Tensor e({2, 3});
    Tensor f({3, 2});
    
    // Fill matrices
    float* e_data = e.data_ptr();
    float* f_data = f.data_ptr();
    for (size_t i = 0; i < 6; ++i) {
        e_data[i] = static_cast<float>(i + 1);
        f_data[i] = static_cast<float>(i + 1);
    }
    
    std::cout << "Matrix e (2x3): ";
    e.print();
    std::cout << "Matrix f (3x2): ";
    f.print();
    
    Tensor matmul_result = matmul(e, f);
    std::cout << "Matrix multiplication result (2x2): ";
    matmul_result.print();
    
    // Test 5: Element-wise operations
    std::cout << "\n5. Testing element-wise operations:" << std::endl;
    Tensor g({3});
    Tensor h({3});
    
    float* g_data = g.data_ptr();
    float* h_data = h.data_ptr();
    for (size_t i = 0; i < 3; ++i) {
        g_data[i] = static_cast<float>(i + 1);
        h_data[i] = static_cast<float>(i + 2);
    }
    
    std::cout << "Tensor g: ";
    g.print();
    std::cout << "Tensor h: ";
    h.print();
    
    Tensor add_result = add(g, h);
    std::cout << "Addition result: ";
    add_result.print();
    
    Tensor mul_result = multiply(g, h);
    std::cout << "Multiplication result: ";
    mul_result.print();
    
    std::cout << "\n=== All tests completed ===" << std::endl;
    
    return 0;
}
