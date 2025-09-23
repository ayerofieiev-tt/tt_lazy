#pragma once
#include "TapeOptimizationPass.hpp"

// MLP fusion pass - fuses MatMul + Add + ReLU patterns
class MLPFusionPass : public TapeOptimizationPass {
public:
    int apply(Tape& tape, const std::vector<Tensor>& outputs) override;
    std::string name() const override { return "MLPFusion"; }
    int priority() const override { return 50; } // Run after dead code elimination
};
