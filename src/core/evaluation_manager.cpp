#include "evaluation_manager.hpp"

namespace tt_lazy {

// Simple stub implementation for demo purposes
class StubEvaluationManager : public EvaluationManager {
public:
    void evaluate(Tensor& tensor) override {
        // For demo purposes, we just mark the tensor as evaluated
        // In a real implementation, this would actually compute the tensor
        // tensor.spec_->state = TensorState::EVALUATED;
        // For now, we'll just do nothing since we're only doing graph visualization
        (void)tensor; // Suppress unused parameter warning
    }
};

EvaluationManager& get_evaluation_manager() {
    static StubEvaluationManager instance;
    return instance;
}

}  // namespace tt_lazy
