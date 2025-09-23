#pragma once
#include "TapeOptimizationPass.hpp"

// Dead code elimination pass - removes unused operations
class DeadCodeEliminationPass : public TapeOptimizationPass {
public:
    int apply(Tape& tape, const std::vector<Tensor>& outputs) override;
    std::string name() const override { return "DeadCodeElimination"; }
    int priority() const override { return 10; } // Run early
};
