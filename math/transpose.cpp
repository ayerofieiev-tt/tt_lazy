#include "Tensor.hpp"
#include <stdexcept>

namespace math {

Tensor transpose(const Tensor& input, const std::vector<int32_t>& dims) {
    if (dims.empty()) {
        // Default: reverse the last two dimensions
        if (input.rank() < 2) {
            throw std::runtime_error("Transpose requires at least 2D tensor");
        }
        
        std::vector<uint32_t> output_shape(input.shape(), input.shape() + input.rank());
        std::swap(output_shape[input.rank() - 2], output_shape[input.rank() - 1]);
        
        Tensor result(output_shape);
        
        // Perform transpose
        uint32_t rows = input.size(input.rank() - 2);
        uint32_t cols = input.size(input.rank() - 1);
        
        const float* input_data = input.const_data_ptr();
        float* result_data = result.data_ptr();
        
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                result_data[j * rows + i] = input_data[i * cols + j];
            }
        }
        
        return result;
    } else {
        throw std::runtime_error("Custom dimension transpose not implemented");
    }
}

} // namespace math
