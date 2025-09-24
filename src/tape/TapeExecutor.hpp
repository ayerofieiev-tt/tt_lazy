#pragma once
#include "Tape.hpp"
#include "TapeOperation.hpp"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

// Forward declarations
class Tape;

// Function signature for operation execution
using OperationHandler = std::function<void(TapeOperation&, TapeExecutor&)>;

// Tape executor - executes tape using registered operation handlers
class TapeExecutor {
   public:
    TapeExecutor() = default;

    // Execute entire tape
    void execute_tape(Tape& tape);

    // Execute single operation
    void execute_operation(TapeOperation& op);

    // Operation registry methods
    void register_operation(OpTypeId op_type, OperationHandler handler);
    bool is_registered(OpTypeId op_type) const;
    size_t get_num_registered_operations() const;

    // Result management
    std::shared_ptr<Tensor> get_result(NodeId node_id) const;
    void set_result(NodeId node_id, std::shared_ptr<Tensor> result);

    // Memory management
    void clear_results();
    size_t memory_usage() const;

   private:
    std::unordered_map<NodeId, std::shared_ptr<Tensor>> results_;
    std::vector<OperationHandler> operation_handlers_;
};

// Global function to register all standard operations with a TapeExecutor
void register_all_operations(TapeExecutor& executor);
