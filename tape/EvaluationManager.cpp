#include "EvaluationManager.hpp"
#include "TapeExecutor.hpp"
#include <algorithm>

EvaluationManager::EvaluationManager() {
    // Register all standard operations with the executor
    register_all_operations(executor_);
}

std::shared_ptr<Tensor> EvaluationManager::evaluate(const Tensor& tensor) {
    if (tensor.is_materialized()) {
        stats_.cache_hits++;
        auto result = std::make_shared<Tensor>(tensor);
        // Ensure the copy is also materialized
        if (!result->is_materialized()) {
            result->materialize();
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

std::vector<std::shared_ptr<Tensor>> EvaluationManager::evaluate(const std::vector<Tensor>& tensors) {
    std::vector<std::shared_ptr<Tensor>> results;
    results.reserve(tensors.size());
    
    for (const auto& tensor : tensors) {
        results.push_back(evaluate(tensor));
    }
    
    return results;
}

bool EvaluationManager::is_evaluated(const Tensor& tensor) const {
    if (tensor.is_materialized()) {
        return true;
    }
    
    if (tensor.is_lazy()) {
        return evaluation_cache_.count(tensor.producer_node()) > 0;
    }
    
    return false;
}

void EvaluationManager::clear_cache() {
    evaluation_cache_.clear();
    executor_.clear_results();
    reset_stats();
}

std::shared_ptr<Tensor> EvaluationManager::evaluate_impl(const Tensor& tensor) {
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

bool EvaluationManager::needs_evaluation(const Tensor& tensor) const {
    if (tensor.is_materialized()) {
        return false;
    }
    
    if (tensor.is_lazy()) {
        return evaluation_cache_.count(tensor.producer_node()) == 0;
    }
    
    return false;
}

// Global evaluation manager instance
EvaluationManager& get_evaluation_manager() {
    static EvaluationManager instance;
    return instance;
}
