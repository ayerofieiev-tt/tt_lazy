#pragma once

#include "tensor.hpp"

#include <memory>

namespace tt_lazy {

/**
 * Abstract interface for tensor evaluation managers.
 * This allows different evaluation strategies (tape-based, immediate, etc.)
 * to be implemented without creating circular dependencies.
 */
class EvaluationManager {
public:
    virtual ~EvaluationManager() = default;

    // Non-copyable, non-movable (abstract base class)
    EvaluationManager(const EvaluationManager&) = delete;
    EvaluationManager& operator=(const EvaluationManager&) = delete;
    EvaluationManager(EvaluationManager&&) = delete;
    EvaluationManager& operator=(EvaluationManager&&) = delete;

    /**
     * Evaluate a lazy tensor in-place, materializing its data.
     * @param tensor The lazy tensor to evaluate (will be modified in-place)
     */
    virtual void evaluate(Tensor& tensor) = 0;

protected:
    EvaluationManager() = default;
};

/**
 * Get the global evaluation manager instance.
 * This will automatically initialize with the tape-based implementation
 * when the tape library is linked.
 */
EvaluationManager& get_evaluation_manager();

}  // namespace tt_lazy
