#include "Tensor.hpp"
#include <algorithm>
#include <stdexcept>

namespace math {

Tensor fused_mlp(const Tensor& input, const Tensor& weights, const Tensor& bias, bool has_relu) {
    // Validate inputs are materialized
    if (!input.is_evaluated() || !weights.is_evaluated() || !bias.is_evaluated()) {
        throw std::runtime_error("Fused MLP requires materialized input tensors");
    }
    
    // Get dimensions
    size_t batch_size = input.size(0);
    size_t input_features = input.size(1);
    size_t output_features = weights.size(1);
    
    // Validate shapes
    if (weights.size(0) != input_features) {
        throw std::runtime_error("Incompatible shapes for MLP: input features don't match weight rows");
    }
    if (bias.size(1) != output_features) {
        throw std::runtime_error("Incompatible shapes for MLP: bias features don't match weight columns");
    }
    
    // Create output tensor
    std::vector<uint32_t> output_shape = {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(output_features)};
    Tensor result(output_shape);
    
    // Get data pointers
    const float* input_data = input.const_data_ptr();
    const float* weights_data = weights.const_data_ptr();
    const float* bias_data = bias.const_data_ptr();
    float* result_data = result.data_ptr();
    
    // Fused computation: MatMul + Add + (optional ReLU)
    // This is more efficient than separate operations
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t out_feat = 0; out_feat < output_features; ++out_feat) {
            // Compute dot product: input[batch, :] @ weights[:, out_feat]
            float sum = 0.0f;
            for (size_t in_feat = 0; in_feat < input_features; ++in_feat) {
                size_t input_idx = batch * input_features + in_feat;
                size_t weight_idx = in_feat * output_features + out_feat;
                sum += input_data[input_idx] * weights_data[weight_idx];
            }
            
            // Add bias
            sum += bias_data[out_feat];  // bias has shape [1, output_features]
            
            // Apply ReLU if requested
            if (has_relu) {
                sum = std::max(0.0f, sum);
            }
            
            // Store result
            size_t result_idx = batch * output_features + out_feat;
            result_data[result_idx] = sum;
        }
    }
    
    return result;
}

} // namespace math
