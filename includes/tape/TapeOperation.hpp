#pragma once
#include "Tensor.hpp"
#include "common.hpp"

#include <memory>
#include <vector>

// Represents a single operation in the execution tape
struct TapeOperation {
    NodeId node_id;
    OpTypeId op_type;
    std::vector<NodeId> input_nodes;      // Dependencies (lazy tensors)
    std::vector<Tensor> constant_inputs;  // Constant input tensors
    std::vector<NodeId> output_nodes;     // Produced tensors
    std::vector<std::vector<uint32_t>> output_shapes;

    // Execution metadata
    bool is_constant = false;
    bool is_evaluated = false;
    std::shared_ptr<Tensor> result;  // Actual computed result

    TapeOperation(
        NodeId node_id,
        OpTypeId
            op_type)  // NOLINT(bugprone-easily-swappable-parameters) - Both are uint32_t but semantically different
        : node_id(node_id), op_type(op_type) {}
};
