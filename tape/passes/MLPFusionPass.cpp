#include "MLPFusionPass.hpp"
#include "Tape.hpp"
#include "operations.hpp"
#include <iostream>
#include <unordered_set>
#include <algorithm>

int MLPFusionPass::apply(Tape& tape, const std::vector<Tensor>& outputs) {
    std::cout << "  🔥 Applying MLP fusion..." << std::endl;
    
    // Access tape operations through base class helper
    auto& operations = get_operations(tape);
    
    // Find MatMul + Add patterns and fuse them
    std::vector<std::unique_ptr<TapeOperation>> new_operations;
    std::unordered_set<size_t> fused_indices;
    int fusions_count = 0;
    
    for (size_t i = 0; i < operations.size(); ++i) {
        if (fused_indices.count(i)) continue;  // Skip already fused operations
        
        auto& current_op = operations[i];
        
        // Look for MatMul operation
        if (current_op->op_type == MatMulArgs::type_id()) {
            // Look for Add operation that uses this MatMul as input
            for (size_t j = i + 1; j < operations.size(); ++j) {
                if (fused_indices.count(j)) continue;
                
                auto& candidate_add = operations[j];
                if (candidate_add->op_type == AddArgs::type_id()) {
                    // Check if this Add uses the MatMul output
                    bool uses_matmul = false;
                    for (NodeId input : candidate_add->input_nodes) {
                        if (input == current_op->node_id) {
                            uses_matmul = true;
                            break;
                        }
                    }
                    
                    if (uses_matmul) {
                        // Found MatMul + Add pattern! Create fused operation
                        std::cout << "    🔗 Fusing MatMul(" << current_op->node_id 
                                  << ") + Add(" << candidate_add->node_id << ") → FusedMLP" << std::endl;
                        
                        // Create new fused operation that replaces both
                        auto fused_op = std::make_unique<TapeOperation>(
                            candidate_add->node_id,  // Use the Add's node_id as the result
                            FusedMLPArgs::type_id()   // Mark as fused MLP operation
                        );
                        
                        // Copy inputs from MatMul (input + weights) 
                        fused_op->input_nodes = current_op->input_nodes;
                        fused_op->constant_inputs = current_op->constant_inputs;
                        
                        // Add bias from Add operation's constant inputs
                        for (const auto& bias : candidate_add->constant_inputs) {
                            fused_op->constant_inputs.push_back(bias);
                        }
                        
                        // Set output
                        fused_op->output_nodes = candidate_add->output_nodes;
                        fused_op->output_shapes = candidate_add->output_shapes;
                        
                        new_operations.push_back(std::move(fused_op));
                        
                        // Mark both operations as fused
                        fused_indices.insert(i);
                        fused_indices.insert(j);
                        fusions_count++;
                        break;  // Found fusion for this MatMul
                    }
                }
            }
        }
        
        // If not fused, keep the original operation
        if (!fused_indices.count(i)) {
            new_operations.push_back(std::move(operations[i]));
        }
    }
    
    // Replace operations with fused version
    operations = std::move(new_operations);
    
    std::cout << "    ✅ Fused " << fusions_count << " MLP patterns" << std::endl;
    return fusions_count;
}
