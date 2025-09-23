#include "Tensor.hpp"
#include <stdexcept>

namespace math {

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b) {
    // Validate input shapes
    if (a.rank() < 2 || b.rank() < 2) {
        throw std::runtime_error("Matrix multiplication requires at least 2D tensors");
    }
    
    // Get matrix dimensions
    uint32_t a_rows = transpose_a ? a.size(a.rank() - 1) : a.size(a.rank() - 2);
    uint32_t a_cols = transpose_a ? a.size(a.rank() - 2) : a.size(a.rank() - 1);
    uint32_t b_rows = transpose_b ? b.size(b.rank() - 1) : b.size(b.rank() - 2);
    uint32_t b_cols = transpose_b ? b.size(b.rank() - 2) : b.size(b.rank() - 1);
    
    if (a_cols != b_rows) {
        throw std::runtime_error("Matrix dimension mismatch for multiplication");
    }
    
    // Calculate output shape
    std::vector<uint32_t> output_shape;
    
    // Handle batch dimensions (if any)
    size_t min_rank = std::min(a.rank(), b.rank());
    for (size_t i = 0; i < min_rank - 2; ++i) {
        output_shape.push_back(std::max(a.size(i), b.size(i)));
    }
    
    // Add matrix dimensions
    output_shape.push_back(a_rows);
    output_shape.push_back(b_cols);
    
    Tensor result(output_shape);
    
    // Perform matrix multiplication
    // This is a simplified implementation for 2D matrices
    if (a.rank() == 2 && b.rank() == 2) {
        const float* a_data = a.const_data_ptr();
        const float* b_data = b.const_data_ptr();
        float* result_data = result.data_ptr();
        
        for (uint32_t i = 0; i < a_rows; ++i) {
            for (uint32_t j = 0; j < b_cols; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < a_cols; ++k) {
                    float a_val = transpose_a ? a_data[k * a_rows + i] : a_data[i * a_cols + k];
                    float b_val = transpose_b ? b_data[j * b_rows + k] : b_data[k * b_cols + j];
                    sum += a_val * b_val;
                }
                result_data[i * b_cols + j] = sum;
            }
        }
    } else {
        // For higher-dimensional tensors, we'd need more complex implementation
        throw std::runtime_error("Multi-dimensional matrix multiplication not fully implemented");
    }
    
    return result;
}

} // namespace math