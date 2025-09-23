#pragma once
#include "TapeOperation.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>

// Forward declarations
class TapeExecutor;

// Execution tape - linear sequence of operations
class Tape {
public:
    Tape() = default;
    
    // Add operation to tape
    void add_operation(std::unique_ptr<TapeOperation> op);
    
    // Get operations in execution order
    const std::vector<std::unique_ptr<TapeOperation>>& operations() const;
    
    // Find operation by node ID
    TapeOperation* find_operation(NodeId node_id);
    const TapeOperation* find_operation(NodeId node_id) const;
    
    // Get all dependencies for a given node
    std::vector<NodeId> get_dependencies(NodeId node_id) const;
    
    // Tape optimization passes
    void eliminate_dead_code(const std::vector<Tensor>& required_outputs);
    void fuse_operations(); // Future: operation fusion
    void reorder_for_memory(); // Future: memory optimization
    
    // Validation
    bool is_valid() const;
    void validate() const;
    
    // Debugging
    void print_tape(std::ostream& os = std::cout) const;
    size_t size() const { return operations_.size(); }
    
private:
    std::vector<std::unique_ptr<TapeOperation>> operations_;
    std::unordered_map<NodeId, TapeOperation*> node_to_op_;
    
    void build_node_map();
};
