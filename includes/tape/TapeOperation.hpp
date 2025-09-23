#pragma once
#include "common.hpp"
#include "Tensor.hpp"
#include <vector>
#include <memory>

// Represents a single operation in the execution tape
struct TapeOperation {
    NodeId node_id;
    OpTypeId op_type;
    std::vector<NodeId> input_nodes;  // Dependencies (lazy tensors)
    std::vector<Tensor> constant_inputs; // Constant input tensors
    std::vector<NodeId> output_nodes; // Produced tensors
    std::vector<std::vector<uint32_t>> output_shapes;
    
    // Execution metadata
    bool is_constant = false;
    bool is_evaluated = false;
    std::shared_ptr<Tensor> result; // Actual computed result
    
    TapeOperation(NodeId id, OpTypeId type) 
        : node_id(id), op_type(type) {}
};
