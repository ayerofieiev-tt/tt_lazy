#pragma once
#include "tensor.hpp"
#include "common.hpp"

#include <vector>

// Operation argument definitions
DEFINE_OP_ARGS(Split, uint32_t split_size = 0; uint32_t dim = 0;);

DEFINE_OP_ARGS(MatMul, bool transpose_a = false; bool transpose_b = false; float alpha = 1.0f; float beta = 0.0f;);

DEFINE_OP_ARGS(Reduce, SmallVector<int32_t, 4> dims; bool keepdim = false; enum class Type
               : uint8_t{SUM, MEAN, MAX, MIN} type = Type::SUM;);

DEFINE_OP_ARGS(ReLU, bool inplace = false;);

DEFINE_OP_ARGS(Add,
               // No additional arguments needed
);

DEFINE_OP_ARGS(Multiply,
               // No additional arguments needed
);

DEFINE_OP_ARGS(FusedMLP,
               // Store the fused MLP parameters
               bool has_relu = true;          // Whether to apply ReLU activation
               std::string fusion_info = "";  // Debug info about what was fused
);

// Operation implementations
std::vector<Tensor> split(const Tensor& input, uint32_t split_size, uint32_t dim = 0);
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false);
Tensor reduce_sum(const Tensor& input, const std::vector<int32_t>& dims = {}, bool keepdim = false);
Tensor relu(const Tensor& input);
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor fused_mlp(const Tensor& input, const Tensor& weights, const Tensor& bias, bool has_relu = true);

// Tensor creation functions (float32 only for now)
Tensor zeros(const Shape& shape);
Tensor ones(const Shape& shape);
Tensor rand(const Shape& shape);