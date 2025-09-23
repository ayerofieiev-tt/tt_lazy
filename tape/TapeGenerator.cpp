#include "TapeGenerator.hpp"
#include "Context.hpp"
#include "Node.hpp"
#include <algorithm>
#include <queue>
#include <unordered_set>

std::unique_ptr<Tape> TapeGenerator::generate_tape(const std::vector<Tensor>& outputs) {
    auto tape = std::make_unique<Tape>();
    
    // Collect all dependencies
    std::vector<NodeId> dependencies = collect_dependencies(outputs);
    
    // Topologically sort dependencies
    std::vector<NodeId> sorted_nodes = topological_sort(dependencies);
    
    // Create tape operations
    auto& context = Context::instance();
    for (NodeId node_id : sorted_nodes) {
        const Node* node = context.get_node(node_id);
        if (node) {
            auto tape_op = create_tape_operation(*node);
            tape->add_operation(std::move(tape_op));
        }
    }
    
    return tape;
}

std::unique_ptr<Tape> TapeGenerator::generate_tape(const Tensor& output) {
    return generate_tape(std::vector<Tensor>{output});
}

std::vector<NodeId> TapeGenerator::collect_dependencies(const std::vector<Tensor>& outputs) {
    std::unordered_set<NodeId> visited;
    std::vector<NodeId> dependencies;
    
    std::function<void(NodeId)> collect = [&](NodeId node_id) {
        if (visited.count(node_id)) return;
        visited.insert(node_id);
        
        const Node* node = Context::instance().get_node(node_id);
        if (node) {
            // Add dependencies first
            for (const auto& input : node->inputs()) {
                if (input.is_lazy()) {
                    collect(input.producer_node());
                }
            }
            dependencies.push_back(node_id);
        }
    };
    
    for (const auto& tensor : outputs) {
        if (tensor.is_lazy()) {
            collect(tensor.producer_node());
        }
    }
    
    return dependencies;
}

std::vector<NodeId> TapeGenerator::topological_sort(const std::vector<NodeId>& nodes) {
    std::unordered_map<NodeId, std::vector<NodeId>> graph;
    std::unordered_map<NodeId, int> in_degree;
    
    // Build graph and calculate in-degrees
    for (NodeId node_id : nodes) {
        const Node* node = Context::instance().get_node(node_id);
        if (node) {
            in_degree[node_id] = 0;
            for (const auto& input : node->inputs()) {
                if (input.is_lazy()) {
                    NodeId input_id = input.producer_node();
                    graph[input_id].push_back(node_id);
                    in_degree[node_id]++;
                }
            }
        }
    }
    
    // Kahn's algorithm for topological sort
    std::queue<NodeId> queue;
    for (const auto& [node_id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(node_id);
        }
    }
    
    std::vector<NodeId> result;
    while (!queue.empty()) {
        NodeId current = queue.front();
        queue.pop();
        result.push_back(current);
        
        for (NodeId neighbor : graph[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }
    
    return result;
}

std::unique_ptr<TapeOperation> TapeGenerator::create_tape_operation(const Node& node) {
    auto op = std::make_unique<TapeOperation>(node.id(), node.type_id());
    
    // Set input dependencies - only include lazy inputs
    // Constant inputs will be handled by the tape executor directly
    for (const auto& input : node.inputs()) {
        if (input.is_lazy()) {
            op->input_nodes.push_back(input.producer_node());
        }
    }
    
    // Store constant inputs separately for the tape executor
    for (const auto& input : node.inputs()) {
        if (input.is_constant()) {
            op->constant_inputs.push_back(input);
        }
    }
    
    // Set output nodes (for now, just the node itself)
    op->output_nodes.push_back(node.id());
    
    // Set output shapes (simplified - would need proper shape inference)
    std::vector<uint32_t> shape = {1, 1, 1, 1}; // Default shape
    op->output_shapes.push_back(shape);
    
    return op;
}
