#include "Tensor.hpp"

#include <stdexcept>

namespace math {

namespace {
struct MatrixDimensions {
    uint32_t rows;
    uint32_t cols;
};

MatrixDimensions get_matrix_dimensions(const Tensor& tensor, bool transpose) {
    if (transpose) {
        return {tensor.size(tensor.rank() - 1), tensor.size(tensor.rank() - 2)};
    }
    return {tensor.size(tensor.rank() - 2), tensor.size(tensor.rank() - 1)};
}

std::vector<uint32_t> calculate_output_shape(const Tensor& a, const Tensor& b, uint32_t a_rows, uint32_t b_cols) {
    size_t min_rank = std::min(a.rank(), b.rank());
    std::vector<uint32_t> output_shape;
    output_shape.reserve(min_rank);

    // Handle batch dimensions (if any)
    for (size_t i = 0; i < min_rank - 2; ++i) {
        output_shape.push_back(std::max(a.size(i), b.size(i)));
    }

    // Add matrix dimensions
    output_shape.push_back(a_rows);
    output_shape.push_back(b_cols);
    return output_shape;
}

void perform_2d_matrix_multiplication(const Tensor& a, const Tensor& b, Tensor& result, bool transpose_a,
                                      bool transpose_b, uint32_t a_rows, uint32_t a_cols, uint32_t b_cols,
                                      uint32_t b_rows) {
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
}
}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b) {
    // Validate input shapes
    if (a.rank() < 2 || b.rank() < 2) {
        throw std::runtime_error("Matrix multiplication requires at least 2D tensors");
    }

    // Get matrix dimensions
    auto a_dims = get_matrix_dimensions(a, transpose_a);
    auto b_dims = get_matrix_dimensions(b, transpose_b);

    if (a_dims.cols != b_dims.rows) {
        throw std::runtime_error("Matrix dimension mismatch for multiplication");
    }

    // Calculate output shape
    auto output_shape = calculate_output_shape(a, b, a_dims.rows, b_dims.cols);
    Tensor result(output_shape);

    // Perform matrix multiplication
    if (a.rank() == 2 && b.rank() == 2) {
        perform_2d_matrix_multiplication(a, b, result, transpose_a, transpose_b, a_dims.rows, a_dims.cols, b_dims.cols,
                                         b_dims.rows);
    } else {
        // For higher-dimensional tensors, we'd need more complex implementation
        throw std::runtime_error("Multi-dimensional matrix multiplication not fully implemented");
    }

    return result;
}

}  // namespace math
