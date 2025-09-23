#pragma once
#include "common.hpp"
#include "Tensor.hpp"
#include "Context.hpp"
#include <vector>

// Operation argument definitions
DEFINE_OP_ARGS(Split,
    int64_t split_size;
    int32_t dim = 0;
);

DEFINE_OP_ARGS(MatMul,
    bool transpose_a = false;
    bool transpose_b = false;
    float alpha = 1.0f;
    float beta = 0.0f;
);

DEFINE_OP_ARGS(Reduce,
    SmallVector<int32_t, 4> dims;
    bool keepdim = false;
    enum class Type : uint8_t { SUM, MEAN, MAX, MIN } type = Type::SUM;
);

DEFINE_OP_ARGS(ReLU,
    bool inplace = false;
);

// Helper functions
std::vector<Tensor> make_output_tensors(NodeId node_id, size_t num_outputs, const std::vector<std::vector<uint32_t>>& shapes);

// Operation implementations
std::vector<Tensor> split(const Tensor& input, int64_t split_size, int32_t dim = 0);
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false);
Tensor reduce_sum(const Tensor& input, const std::vector<int32_t>& dims = {}, bool keepdim = false);
Tensor relu(const Tensor& input);

