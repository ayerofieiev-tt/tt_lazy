#pragma once

#include "Tensor.hpp"
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
    
    /**
     * Evaluate a lazy tensor and return a materialized result.
     * @param tensor The lazy tensor to evaluate
     * @return A shared pointer to the materialized tensor
     */
    virtual std::shared_ptr<Tensor> evaluate(const Tensor& tensor) = 0;
    
    /**
     * Clear any cached evaluation results.
     */
    virtual void clear_cache() = 0;
    
    /**
     * Evaluation statistics structure.
     */
    struct EvaluationStats {
        size_t cache_hits = 0;
        size_t cache_misses = 0;
        size_t operations_executed = 0;
        size_t memory_allocated = 0;
    };

    /**
     * Get evaluation statistics (optional, can return empty struct).
     */
    virtual EvaluationStats get_stats() const = 0;
};

/**
 * Get the global evaluation manager instance.
 * This will automatically initialize with the tape-based implementation
 * when the tape library is linked.
 */
EvaluationManager& get_evaluation_manager();

} // namespace tt_lazy
