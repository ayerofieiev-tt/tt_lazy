#pragma once
#include "TapeOptimizationPass.hpp"

// MLP fusion pass - fuses MatMul + Add + ReLU patterns
class MLPFusionPass : public TapeOptimizationPass {
   public:
    int apply(Tape& tape, const std::vector<Tensor>& outputs) override;
    std::string name() const override { return "MLPFusion"; }
    static constexpr int MLP_FUSION_PRIORITY = 50;
    int priority() const override { return MLP_FUSION_PRIORITY; }
};
