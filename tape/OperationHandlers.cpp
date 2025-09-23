#include "TapeExecutor.hpp"
#include "operations.hpp"
#include "math_operations.hpp"
#include <stdexcept>


// Operation handler implementations
namespace {
void handle_split(TapeOperation& op, TapeExecutor& executor) {
    // Collect all input tensors (both lazy and constant)
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    
    // Add lazy input tensors
    for (NodeId node_id : op.input_nodes) {
        auto tensor = executor.get_result(node_id);
        if (!tensor) {
            throw std::runtime_error("Missing lazy input tensor for split operation");
        }
        input_tensors.push_back(tensor);
    }
    
    // Add constant input tensors
    for (const auto& const_tensor : op.constant_inputs) {
        input_tensors.push_back(std::make_shared<Tensor>(const_tensor));
    }
    
    if (input_tensors.size() != 1) {
        throw std::runtime_error("Split operation requires exactly 1 input, got " + std::to_string(input_tensors.size()));
    }
    
    // For now, create a simple split (this would need proper parameters)
    auto result = std::make_shared<Tensor>(*input_tensors[0]); // Simplified
    executor.set_result(op.node_id, result);
    op.result = result;
}

void handle_matmul(TapeOperation& op, TapeExecutor& executor) {
    // Collect all input tensors (both lazy and constant)
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    
    // Add lazy input tensors
    for (NodeId node_id : op.input_nodes) {
        auto tensor = executor.get_result(node_id);
        if (!tensor) {
            throw std::runtime_error("Missing lazy input tensor for matmul operation");
        }
        input_tensors.push_back(tensor);
    }
    
    // Add constant input tensors
    for (const auto& const_tensor : op.constant_inputs) {
        input_tensors.push_back(std::make_shared<Tensor>(const_tensor));
    }
    
    if (input_tensors.size() != 2) {
        throw std::runtime_error("MatMul operation requires exactly 2 inputs, got " + std::to_string(input_tensors.size()));
    }
    
    // Call math function
    auto result = std::make_shared<Tensor>(math::matmul(*input_tensors[0], *input_tensors[1]));
    executor.set_result(op.node_id, result);
    op.result = result;
}

void handle_reduce(TapeOperation& op, TapeExecutor& executor) {
    // Collect all input tensors (both lazy and constant)
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    
    // Add lazy input tensors
    for (NodeId node_id : op.input_nodes) {
        auto tensor = executor.get_result(node_id);
        if (!tensor) {
            throw std::runtime_error("Missing lazy input tensor for reduce operation");
        }
        input_tensors.push_back(tensor);
    }
    
    // Add constant input tensors
    for (const auto& const_tensor : op.constant_inputs) {
        input_tensors.push_back(std::make_shared<Tensor>(const_tensor));
    }
    
    if (input_tensors.size() != 1) {
        throw std::runtime_error("Reduce operation requires exactly 1 input, got " + std::to_string(input_tensors.size()));
    }
    
    // Call math function (simplified - would need proper parameters)
    std::vector<int32_t> dims = {0}; // Default: reduce along first dimension
    auto result = std::make_shared<Tensor>(math::reduce_sum(*input_tensors[0], dims));
    executor.set_result(op.node_id, result);
    op.result = result;
}

void handle_relu(TapeOperation& op, TapeExecutor& executor) {
    // Collect all input tensors (both lazy and constant)
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    
    // Add lazy input tensors
    for (NodeId node_id : op.input_nodes) {
        auto tensor = executor.get_result(node_id);
        if (!tensor) {
            throw std::runtime_error("Missing lazy input tensor for relu operation");
        }
        input_tensors.push_back(tensor);
    }
    
    // Add constant input tensors
    for (const auto& const_tensor : op.constant_inputs) {
        input_tensors.push_back(std::make_shared<Tensor>(const_tensor));
    }
    
    if (input_tensors.size() != 1) {
        throw std::runtime_error("ReLU operation requires exactly 1 input, got " + std::to_string(input_tensors.size()));
    }
    
    // Call math function
    auto result = std::make_shared<Tensor>(math::relu(*input_tensors[0]));
    executor.set_result(op.node_id, result);
    op.result = result;
}

void handle_add(TapeOperation& op, TapeExecutor& executor) {
    // Collect all input tensors (both lazy and constant)
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    
    // Add lazy input tensors
    for (NodeId node_id : op.input_nodes) {
        auto tensor = executor.get_result(node_id);
        if (!tensor) {
            throw std::runtime_error("Missing lazy input tensor for add operation");
        }
        input_tensors.push_back(tensor);
    }
    
    // Add constant input tensors
    for (const auto& const_tensor : op.constant_inputs) {
        input_tensors.push_back(std::make_shared<Tensor>(const_tensor));
    }
    
    if (input_tensors.size() != 2) {
        throw std::runtime_error("Add operation requires exactly 2 inputs, got " + std::to_string(input_tensors.size()));
    }
    
    // Call math function
    auto result = std::make_shared<Tensor>(math::add(*input_tensors[0], *input_tensors[1]));
    executor.set_result(op.node_id, result);
    op.result = result;
}

void handle_multiply(TapeOperation& op, TapeExecutor& executor) {
    // Collect all input tensors (both lazy and constant)
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    
    // Add lazy input tensors
    for (NodeId node_id : op.input_nodes) {
        auto tensor = executor.get_result(node_id);
        if (!tensor) {
            throw std::runtime_error("Missing lazy input tensor for multiply operation");
        }
        input_tensors.push_back(tensor);
    }
    
    // Add constant input tensors
    for (const auto& const_tensor : op.constant_inputs) {
        input_tensors.push_back(std::make_shared<Tensor>(const_tensor));
    }
    
    if (input_tensors.size() != 2) {
        throw std::runtime_error("Multiply operation requires exactly 2 inputs, got " + std::to_string(input_tensors.size()));
    }
    
    // Call math function
    auto result = std::make_shared<Tensor>(math::multiply(*input_tensors[0], *input_tensors[1]));
    executor.set_result(op.node_id, result);
    op.result = result;
}

void handle_fused_mlp(TapeOperation& op, TapeExecutor& executor) {
    // Collect all input tensors (both lazy and constant)
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    
    // Add lazy input tensors
    for (NodeId node_id : op.input_nodes) {
        auto tensor = executor.get_result(node_id);
        if (!tensor) {
            throw std::runtime_error("Missing lazy input tensor for fused MLP operation");
        }
        input_tensors.push_back(tensor);
    }
    
    // Add constant input tensors
    for (const auto& const_tensor : op.constant_inputs) {
        input_tensors.push_back(std::make_shared<Tensor>(const_tensor));
    }
    
    if (input_tensors.size() != 3) {
        throw std::runtime_error("Fused MLP operation requires exactly 3 inputs (input, weights, bias), got " + std::to_string(input_tensors.size()));
    }
    
    // Get the operation arguments from the tape operation
    auto& ctx = Context::instance();
    const Node* node = ctx.get_node(op.node_id);
    if (!node) {
        throw std::runtime_error("Cannot find node for fused MLP operation");
    }
    
    const auto& args = node->as<FusedMLPArgs>();
    bool has_relu = args.has_relu;
    
    // Call fused math function with input, weights, bias
    auto result = std::make_shared<Tensor>(math::fused_mlp(*input_tensors[0], *input_tensors[1], *input_tensors[2], has_relu));
    executor.set_result(op.node_id, result);
    op.result = result;
}
}

// Global function to register all operations with any TapeExecutor
void register_all_operations(TapeExecutor& executor) {
    executor.register_operation(SplitArgs::type_id(), handle_split);
    executor.register_operation(MatMulArgs::type_id(), handle_matmul);
    executor.register_operation(ReduceArgs::type_id(), handle_reduce);
    executor.register_operation(ReLUArgs::type_id(), handle_relu);
    executor.register_operation(AddArgs::type_id(), handle_add);
    executor.register_operation(MultiplyArgs::type_id(), handle_multiply);
    executor.register_operation(FusedMLPArgs::type_id(), handle_fused_mlp);
}