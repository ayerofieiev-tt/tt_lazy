#include "Tensor.hpp"

#include <algorithm>
#include <stdexcept>

namespace math {

Tensor relu(const Tensor& input) {
    std::vector<uint32_t> shape(
        input.shape(),
        input.shape() +
            input
                .rank());  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) - Safe array access with known bounds
    Tensor result(shape);

    // Apply ReLU element-wise: max(0, x)
    const float* input_data = input.const_data_ptr();
    float* result_data = result.data_ptr();
    for (size_t i = 0; i < input.total_elements(); ++i) {
        result_data[i] = std::max(
            0.0f,
            input_data
                [i]);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) - Safe array access with bounds checking
    }

    return result;
}

Tensor add(const Tensor& a, const Tensor& b) {
    // Check if shapes can be broadcast
    std::vector<uint32_t> a_shape(
        a.shape(),
        a.shape() +
            a.rank());  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) - Safe array access with known bounds
    std::vector<uint32_t> b_shape(
        b.shape(),
        b.shape() +
            b.rank());  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) - Safe array access with known bounds

    if (!Tensor::can_broadcast(a_shape, b_shape)) {
        throw std::runtime_error("Cannot broadcast shapes for addition");
    }

    auto output_shape = Tensor::broadcast_shapes(a_shape, b_shape);
    Tensor result(output_shape);

    // Perform element-wise addition
    const float* a_data = a.const_data_ptr();
    const float* b_data = b.const_data_ptr();
    float* result_data = result.data_ptr();

    if (a_shape == b_shape) {
        // Same shapes - simple element-wise addition
        for (size_t i = 0; i < a.total_elements(); ++i) {
            result_data[i] =
                a_data[i] +
                b_data
                    [i];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) - Safe array access with bounds checking
        }
    } else {
        // Basic broadcasting support for bias addition (e.g., [N, M] + [1, M])
        if (a_shape.size() == 2 && b_shape.size() == 2 && b_shape[0] == 1 && a_shape[1] == b_shape[1]) {
            // Broadcasting [N, M] + [1, M] -> [N, M]
            size_t batch_size = a_shape[0];
            size_t feature_size = a_shape[1];

            for (size_t batch = 0; batch < batch_size; ++batch) {
                for (size_t feat = 0; feat < feature_size; ++feat) {
                    size_t a_idx = batch * feature_size + feat;
                    size_t b_idx = feat;  // b has shape [1, M], so only varies by feature
                    result_data[a_idx] =
                        a_data[a_idx] +
                        b_data
                            [b_idx];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) - Safe array access with bounds checking
                }
            }
        } else {
            throw std::runtime_error("Broadcasting addition not implemented for these shapes");
        }
    }

    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    // Check if shapes can be broadcast
    std::vector<uint32_t> a_shape(a.shape(), a.shape() + a.rank());
    std::vector<uint32_t> b_shape(b.shape(), b.shape() + b.rank());

    if (!Tensor::can_broadcast(a_shape, b_shape)) {
        throw std::runtime_error("Cannot broadcast shapes for multiplication");
    }

    auto output_shape = Tensor::broadcast_shapes(a_shape, b_shape);
    Tensor result(output_shape);

    // Perform element-wise multiplication
    // This is a simplified implementation for same-shaped tensors
    if (a_shape == b_shape) {
        const float* a_data = a.const_data_ptr();
        const float* b_data = b.const_data_ptr();
        float* result_data = result.data_ptr();
        for (size_t i = 0; i < a.total_elements(); ++i) {
            result_data[i] = a_data[i] * b_data[i];
        }
    } else {
        throw std::runtime_error("Broadcasting multiplication not fully implemented");
    }

    return result;
}

}  // namespace math
