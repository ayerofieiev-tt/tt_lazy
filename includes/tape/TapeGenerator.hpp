#pragma once
#include "Tape.hpp"
#include "Tensor.hpp"
#include <vector>
#include <memory>

// Forward declarations
class Node;
class TapeOptimizationPass;

// Tape generator - converts graph to execution tape
class TapeGenerator {
public:
    TapeGenerator() : optimization_enabled_(true) {}
    
    // Generate tape from a set of output tensors
    std::unique_ptr<Tape> generate_tape(const std::vector<Tensor>& outputs);
    
    // Generate tape from single tensor
    std::unique_ptr<Tape> generate_tape(const Tensor& output);
    
    // Control optimization
    void set_optimization_enabled(bool enabled) { optimization_enabled_ = enabled; }
    bool is_optimization_enabled() const { return optimization_enabled_; }
    
    // Optimization pass registry
    static void register_optimization_pass(std::unique_ptr<TapeOptimizationPass> pass);
    static void register_default_passes();
    static void clear_passes();
    
private:
    // Helper methods
    std::vector<NodeId> collect_dependencies(const std::vector<Tensor>& outputs);
    std::vector<NodeId> topological_sort(const std::vector<NodeId>& nodes);
    std::unique_ptr<TapeOperation> create_tape_operation(const Node& node);
    
    // Optimization control
    bool optimization_enabled_ = false;
    
    // Static optimization pass registry
    static std::vector<std::unique_ptr<TapeOptimizationPass>> optimization_passes_;
    static bool default_passes_registered_;
};
