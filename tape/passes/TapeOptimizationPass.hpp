#pragma once
#include "TapeOperation.hpp"
#include "Tensor.hpp"

#include <memory>
#include <string>
#include <vector>

// Forward declarations
class Tape;
class TapeGenerator;

// Base class for tape optimization passes (internal to TapeGenerator)
class TapeOptimizationPass {
   public:
    virtual ~TapeOptimizationPass() = default;

    // Non-copyable, non-movable (abstract base class)
    TapeOptimizationPass(const TapeOptimizationPass&) = delete;
    TapeOptimizationPass& operator=(const TapeOptimizationPass&) = delete;
    TapeOptimizationPass(TapeOptimizationPass&&) = delete;
    TapeOptimizationPass& operator=(TapeOptimizationPass&&) = delete;

   public:
    // Apply optimization during tape generation
    // Returns number of optimizations applied
    virtual int apply(Tape& tape, const std::vector<Tensor>& outputs) = 0;

    // Get pass name for debugging
    virtual std::string name() const = 0;

    // Priority for pass ordering (lower runs first)
    static constexpr int DEFAULT_PRIORITY = 100;
    virtual int priority() const { return DEFAULT_PRIORITY; }

   protected:
    TapeOptimizationPass() = default;
    // Helper to access tape internals through friend access
    std::vector<std::unique_ptr<TapeOperation>>& get_operations(Tape& tape);
    void rebuild_node_map(Tape& tape);
};
