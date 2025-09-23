#include "Tensor.hpp"
#include <algorithm>
#include <stdexcept>

namespace math {

std::vector<Tensor> split(const Tensor& input, int64_t split_size, int32_t dim) {
    if (dim < 0 || dim >= static_cast<int32_t>(input.rank())) {
        throw std::runtime_error("Invalid dimension for split operation");
    }
    
    if (split_size <= 0) {
        throw std::runtime_error("Split size must be positive");
    }
    
    size_t input_size = input.size(dim);
    size_t num_outputs = (input_size + split_size - 1) / split_size;
    
    std::vector<Tensor> outputs;
    outputs.reserve(num_outputs);
    
    // Calculate output shapes
    std::vector<uint32_t> base_shape(input.shape(), input.shape() + input.rank());
    
    for (size_t i = 0; i < num_outputs; ++i) {
        std::vector<uint32_t> output_shape = base_shape;
        int64_t remaining = input_size - i * split_size;
        output_shape[dim] = std::min(split_size, remaining);
        
        Tensor output(output_shape);
        
        // Copy data slice
        // This is a simplified implementation - in practice you'd need
        // more sophisticated indexing for multi-dimensional tensors
        size_t start_idx = i * split_size;
        size_t end_idx = std::min(start_idx + static_cast<size_t>(split_size), input_size);
        size_t slice_size = end_idx - start_idx;
        
        // For now, assume 1D tensors or handle the general case
        if (input.rank() == 1) {
            const float* input_data = input.const_data_ptr();
            float* output_data = output.data_ptr();
            std::copy(input_data + start_idx, 
                     input_data + end_idx, 
                     output_data);
        } else {
            // For multi-dimensional tensors, we'd need more complex indexing
            // This is a placeholder - full implementation would require
            // proper multi-dimensional indexing
            throw std::runtime_error("Multi-dimensional split not fully implemented");
        }
        
        outputs.push_back(std::move(output));
    }
    
    return outputs;
}

} // namespace math