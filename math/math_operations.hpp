#pragma once
#include "Tensor.hpp"
#include <vector>

namespace math {

// Math-based operations that work on actual data
// These perform immediate computation rather than building graphs

// Split operation - splits tensor along a dimension
std::vector<Tensor> split(const Tensor& input, int64_t split_size, int32_t dim = 0);

// Matrix multiplication - performs actual matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false);

// Reduce sum - sums along specified dimensions
Tensor reduce_sum(const Tensor& input, const std::vector<int32_t>& dims = {}, bool keepdim = false);

// ReLU activation - applies ReLU function element-wise
Tensor relu(const Tensor& input);

// Additional utility operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& input, const std::vector<int32_t>& dims = {});

// Fused operations for better performance
Tensor fused_mlp(const Tensor& input, const Tensor& weights, const Tensor& bias, bool has_relu = true);

} // namespace math
