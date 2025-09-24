#pragma once
#include "TapeOperation.hpp"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

// Forward declarations
class TapeOptimizationPass;
class TapeExecutor;

// Execution tape - linear sequence of operations
class Tape {
   public:
    Tape() = default;

    // Add operation to tape
    void add_operation(std::unique_ptr<TapeOperation> op);

    // Get operations in execution order
    const std::vector<std::unique_ptr<TapeOperation>>& operations() const;

    // Allow optimization framework to access internals
    friend class TapeOptimizationPass;
    friend class TapeGenerator;

    // Find operation by node ID
    TapeOperation* find_operation(NodeId node_id);
    const TapeOperation* find_operation(NodeId node_id) const;

    // Get all dependencies for a given node
    std::vector<NodeId> get_dependencies(NodeId node_id) const;

    // Validation
    bool is_valid() const;
    void validate() const;

    // Debugging
    void print_tape(std::ostream& os) const;
    void print_tape() const;  // Uses spdlog for output
    size_t size() const { return operations_.size(); }

   private:
    std::vector<std::unique_ptr<TapeOperation>> operations_;
    std::unordered_map<NodeId, TapeOperation*> node_to_op_;

    void build_node_map();
};
