#pragma once
#include "Tape.hpp"
#include "Tensor.hpp"
#include <vector>
#include <memory>

// Forward declarations
class Node;

// Tape generator - converts graph to execution tape
class TapeGenerator {
public:
    TapeGenerator() = default;
    
    // Generate tape from a set of output tensors
    std::unique_ptr<Tape> generate_tape(const std::vector<Tensor>& outputs);
    
    // Generate tape from single tensor
    std::unique_ptr<Tape> generate_tape(const Tensor& output);
    
private:
    // Helper methods
    std::vector<NodeId> collect_dependencies(const std::vector<Tensor>& outputs);
    std::vector<NodeId> topological_sort(const std::vector<NodeId>& nodes);
    std::unique_ptr<TapeOperation> create_tape_operation(const Node& node);
};
