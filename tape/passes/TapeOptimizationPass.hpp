#pragma once
#include "Tensor.hpp"
#include "tape/TapeOperation.hpp"
#include <string>
#include <vector>
#include <memory>

// Forward declarations
class Tape;
class TapeGenerator;

// Base class for tape optimization passes (internal to TapeGenerator)
class TapeOptimizationPass {
public:
    virtual ~TapeOptimizationPass() = default;
    
    // Apply optimization during tape generation
    // Returns number of optimizations applied
    virtual int apply(Tape& tape, const std::vector<Tensor>& outputs) = 0;
    
    // Get pass name for debugging
    virtual std::string name() const = 0;
    
    // Priority for pass ordering (lower runs first)
    virtual int priority() const { return 100; }
    
protected:
    // Helper to access tape internals through friend access
    std::vector<std::unique_ptr<TapeOperation>>& get_operations(Tape& tape);
    void rebuild_node_map(Tape& tape);
};