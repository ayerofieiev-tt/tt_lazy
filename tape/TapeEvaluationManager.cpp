#include "TapeEvaluationManager.hpp"
#include "TapeGenerator.hpp"
#include "TapeExecutor.hpp"
#include "Context.hpp"
#include "Node.hpp"
#include <algorithm>
#include <stdexcept>

namespace tt_lazy {

TapeEvaluationManager::TapeEvaluationManager() {
    // Register all standard operations with the executor
    register_all_operations(executor_);
}

std::shared_ptr<Tensor> TapeEvaluationManager::evaluate(const Tensor& tensor) {
    if (tensor.is_evaluated()) {
        stats_.cache_hits++;
        auto result = std::make_shared<Tensor>(tensor);
        // Ensure the copy is also materialized
        if (!result->is_evaluated()) {
            result->eval();
        }
        return result;
    }

    // Check if we have a cached result
    if (tensor.is_lazy() && evaluation_cache_.count(tensor.producer_node()) > 0) {
        stats_.cache_hits++;
        return evaluation_cache_[tensor.producer_node()];
    }

    stats_.cache_misses++;
    return evaluate_impl(tensor);
}

void TapeEvaluationManager::clear_cache() {
    evaluation_cache_.clear();
    stats_ = EvaluationManager::EvaluationStats{};
}

EvaluationManager::EvaluationStats TapeEvaluationManager::get_stats() const {
    return stats_;
}

std::shared_ptr<Tensor> TapeEvaluationManager::evaluate_impl(const Tensor& tensor) {
    if (!needs_evaluation(tensor)) {
        return std::make_shared<Tensor>(tensor);
    }

    // Generate tape for this tensor
    auto tape = generator_.generate_tape(tensor);

    // Execute tape
    executor_.execute_tape(*tape);

    // Cache all results from the tape execution
    for (const auto& op : tape->operations()) {
        auto op_result = executor_.get_result(op->node_id);
        if (op_result) {
            evaluation_cache_[op->node_id] = op_result;
            stats_.operations_executed++;
            stats_.memory_allocated += op_result->total_elements() * sizeof(float);
        }
    }

    // Get the final result
    std::shared_ptr<Tensor> result = executor_.get_result(tensor.producer_node());

    return result;
}

bool TapeEvaluationManager::needs_evaluation(const Tensor& tensor) const {
    return tensor.is_lazy() && !tensor.is_evaluated();
}

// Implementation of the global evaluation manager function
// This will be found by the linker when the tape library is linked

EvaluationManager& get_evaluation_manager() {
    static std::unique_ptr<EvaluationManager> instance = std::make_unique<TapeEvaluationManager>();
    return *instance;
}

} // namespace tt_lazy
