#include "TapeExecutor.hpp"
#include "Tape.hpp"
#include <algorithm>
#include <stdexcept>

void TapeExecutor::execute_tape(Tape& tape) {
    for (const auto& op : tape.operations()) {
        execute_operation(*op, tape);
    }
}

void TapeExecutor::execute_operation(TapeOperation& op, Tape& tape) {
    // Check if already executed
    if (op.is_evaluated) {
        return;
    }
    
    // Check if operation type is registered
    if (op.op_type >= operation_handlers_.size() || !operation_handlers_[op.op_type]) {
        throw std::runtime_error("Unknown operation type: " + std::to_string(op.op_type));
    }
    
    // Execute the registered handler
    operation_handlers_[op.op_type](op, *this);
    op.is_evaluated = true;
}

std::shared_ptr<Tensor> TapeExecutor::get_result(NodeId node_id) const {
    auto it = results_.find(node_id);
    return it != results_.end() ? it->second : nullptr;
}

void TapeExecutor::set_result(NodeId node_id, std::shared_ptr<Tensor> result) {
    results_[node_id] = std::move(result);
}

void TapeExecutor::register_operation(OpTypeId op_type, OperationHandler handler) {
    // Resize vector if needed to accommodate the operation type
    if (op_type >= operation_handlers_.size()) {
        operation_handlers_.resize(op_type + 1);
    }
    operation_handlers_[op_type] = std::move(handler);
}

bool TapeExecutor::is_registered(OpTypeId op_type) const {
    return op_type < operation_handlers_.size() && operation_handlers_[op_type] != nullptr;
}

size_t TapeExecutor::get_num_registered_operations() const {
    return operation_handlers_.size();
}

void TapeExecutor::clear_results() {
    results_.clear();
}

size_t TapeExecutor::memory_usage() const {
    size_t total = 0;
    for (const auto& [node_id, tensor] : results_) {
        if (tensor) {
            total += tensor->total_elements() * sizeof(float); // Simplified
        }
    }
    return total;
}
