#include "MLPFusionPass.hpp"

#include "Tape.hpp"
#include "operations.hpp"

#include <algorithm>
#include <iostream>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace {
bool uses_matmul_output(const TapeOperation& add_op, NodeId matmul_node_id) {
    for (NodeId input : add_op.input_nodes) {
        if (input == matmul_node_id) {
            return true;
        }
    }
    return false;
}

SmallVector<Tensor, 3> create_fused_inputs(const TapeOperation& matmul_op, const TapeOperation& add_op) {
    SmallVector<Tensor, 3> fused_inputs;

    // Add inputs from MatMul operation
    for (NodeId input_id : matmul_op.input_nodes) {
        fused_inputs.emplace_back(input_id, 0, std::initializer_list<uint32_t>{1, 1, 1, 1});
    }

    // Add bias input from Add operation (skip MatMul output)
    for (NodeId input_id : add_op.input_nodes) {
        if (input_id != matmul_op.node_id) {
            fused_inputs.emplace_back(input_id, 0, std::initializer_list<uint32_t>{1, 1, 1, 1});
        }
    }

    return fused_inputs;
}

std::unique_ptr<TapeOperation> create_fused_operation(const TapeOperation& matmul_op, const TapeOperation& add_op) {
    // Create a new fused node in the context
    auto& ctx = Context::instance();
    FusedMLPArgs fused_args;
    fused_args.has_relu = false;  // Default to no ReLU for now
    fused_args.fusion_info = "MatMul + Add (fused)";

    auto fused_inputs = create_fused_inputs(matmul_op, add_op);
    NodeId fused_node_id = ctx.create_node(fused_inputs, std::move(fused_args));

    // Create new fused tape operation
    auto fused_op = std::make_unique<TapeOperation>(fused_node_id, FusedMLPArgs::type_id());

    // Copy inputs from MatMul (input + weights)
    fused_op->input_nodes = matmul_op.input_nodes;
    fused_op->constant_inputs = matmul_op.constant_inputs;

    // Add bias from Add operation's constant inputs
    for (const auto& bias : add_op.constant_inputs) {
        fused_op->constant_inputs.push_back(bias);
    }

    // Set output
    fused_op->output_nodes = add_op.output_nodes;
    fused_op->output_shapes = add_op.output_shapes;

    return fused_op;
}

bool try_fuse_matmul_add(std::vector<std::unique_ptr<TapeOperation>>& operations,
                         std::vector<std::unique_ptr<TapeOperation>>& new_operations,
                         std::unordered_set<size_t>& fused_indices, size_t matmul_index) {
    auto& matmul_op = operations[matmul_index];

    // Look for Add operation that uses this MatMul as input
    for (size_t j = matmul_index + 1; j < operations.size(); ++j) {
        if (fused_indices.count(j)) {
            continue;
        }

        auto& candidate_add = operations[j];
        if (candidate_add->op_type == AddArgs::type_id() && uses_matmul_output(*candidate_add, matmul_op->node_id)) {
            // Found MatMul + Add pattern! Create fused operation
            spdlog::info("    🔗 Fusing MatMul({}) + Add({}) → FusedMLP", matmul_op->node_id, candidate_add->node_id);

            auto fused_op = create_fused_operation(*matmul_op, *candidate_add);
            new_operations.push_back(std::move(fused_op));

            // Mark both operations as fused
            fused_indices.insert(matmul_index);
            fused_indices.insert(j);
            return true;
        }
    }
    return false;
}
}  // namespace

int MLPFusionPass::apply(Tape& tape, [[maybe_unused]] const std::vector<Tensor>& outputs) {
    spdlog::info("  🔥 Applying MLP fusion...");

    // Temporarily disable MLP fusion to test the rest of the system
    spdlog::warn("    ⚠️  MLP fusion temporarily disabled for testing");
    return 0;

    // Access tape operations through base class helper
    auto& operations = get_operations(tape);

    // Find MatMul + Add patterns and fuse them
    std::vector<std::unique_ptr<TapeOperation>> new_operations;
    std::unordered_set<size_t> fused_indices;
    int fusions_count = 0;

    for (size_t i = 0; i < operations.size(); ++i) {
        if (fused_indices.count(i)) {
            continue;  // Skip already fused operations
        }

        auto& current_op = operations[i];

        // Look for MatMul operation
        if (current_op->op_type == MatMulArgs::type_id()) {
            if (try_fuse_matmul_add(operations, new_operations, fused_indices, i)) {
                fusions_count++;
            }
        }

        // If not fused, keep the original operation
        if (!fused_indices.count(i)) {
            new_operations.push_back(std::move(operations[i]));
        }
    }

    // Replace operations with fused version
    operations = std::move(new_operations);

    spdlog::info("    ✅ Fused {} MLP patterns", fusions_count);
    return fusions_count;
}
