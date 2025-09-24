#pragma once

#include "EvaluationManager.hpp"
#include "TapeExecutor.hpp"
#include "TapeGenerator.hpp"

#include <unordered_map>

namespace tt_lazy {

/**
 * Tape-based implementation of the EvaluationManager interface.
 * This provides lazy evaluation using tape generation and execution.
 */
class TapeEvaluationManager : public EvaluationManager {
   public:
    TapeEvaluationManager();
    ~TapeEvaluationManager() override = default;

    // Non-copyable, non-movable (inherits from base class)
    TapeEvaluationManager(const TapeEvaluationManager&) = delete;
    TapeEvaluationManager& operator=(const TapeEvaluationManager&) = delete;
    TapeEvaluationManager(TapeEvaluationManager&&) = delete;
    TapeEvaluationManager& operator=(TapeEvaluationManager&&) = delete;

    std::shared_ptr<Tensor> evaluate(const Tensor& tensor) override;
    void clear_cache() override;
    EvaluationManager::EvaluationStats get_stats() const override;

   private:
    std::shared_ptr<Tensor> evaluate_impl(const Tensor& tensor);
    bool needs_evaluation(const Tensor& tensor) const;

    TapeGenerator generator_;
    TapeExecutor executor_;
    std::unordered_map<NodeId, std::shared_ptr<Tensor>> evaluation_cache_;
    EvaluationManager::EvaluationStats stats_;
};

}  // namespace tt_lazy
