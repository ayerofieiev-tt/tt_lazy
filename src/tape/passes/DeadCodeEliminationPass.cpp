#include "DeadCodeEliminationPass.hpp"

#include "Tape.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <unordered_set>

#include <spdlog/spdlog.h>

int DeadCodeEliminationPass::apply(Tape& tape, const std::vector<Tensor>& outputs) {
    spdlog::info("  🗑️  Applying dead code elimination...");

    // Access tape operations through base class helper
    auto& operations = get_operations(tape);
    size_t original_size = operations.size();

    std::unordered_set<NodeId> required_nodes;

    // Collect all required nodes by traversing backwards from outputs
    std::function<void(NodeId)> collect_required = [&](NodeId node_id) {
        if (required_nodes.count(node_id))
            return;
        required_nodes.insert(node_id);

        // Find the operation for this node
        for (const auto& op : operations) {
            if (op->node_id == node_id) {
                for (NodeId input : op->input_nodes) {
                    collect_required(input);
                }
                break;
            }
        }
    };

    // Start from required outputs
    for (const auto& tensor : outputs) {
        if (tensor.is_lazy()) {
            collect_required(tensor.producer_node());
        }
    }

    // Remove operations that are not required
    operations.erase(std::remove_if(operations.begin(), operations.end(),
                                    [&](const std::unique_ptr<TapeOperation>& op) {
                                        return required_nodes.count(op->node_id) == 0;
                                    }),
                     operations.end());

    // Rebuild node map after modification
    rebuild_node_map(tape);

    size_t eliminated = original_size - operations.size();
    spdlog::info("    ✅ Eliminated {} dead operations", eliminated);
    return static_cast<int>(eliminated);
}
