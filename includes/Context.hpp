#pragma once
#include "common.hpp"
#include "Tensor.hpp"
#include "Node.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>

// Global context for graph building
class Context {
public:
    Context();
        
    Node* get_node(NodeId id);
    const Node* get_node(NodeId id) const;
    
    // Get all nodes for inspection
    const std::vector<Node>& get_all_nodes() const;
    std::vector<Node>& get_all_nodes();
    
    // Build dependency graph from output tensors
    std::unordered_set<NodeId> get_dependencies(const std::vector<Tensor>& outputs) const;
    
    // Topological sort for execution
    std::vector<NodeId> topological_sort(const std::unordered_set<NodeId>& node_set) const;
    
    size_t size() const;
    void clear();
    
    void print_stats() const;
    
    static Context& instance();

    template<typename ArgsT>
    NodeId create_node(const SmallVector<Tensor, 2>& inputs, ArgsT&& args) {
        NodeId id = next_id_++;
        
        size_t index = nodes_.size();
        nodes_.emplace_back(id, inputs, std::forward<ArgsT>(args));
        id_to_index_[id] = index;
        
        // Update connectivity for input nodes
        for (const auto& input : inputs) {
            if (!input.is_constant() && input.producer_node() != 0) {
                if (Node* producer = get_node(input.producer_node())) {
                    producer->add_output_node(id);
                }
            }
        }
        
        return id;
    }
    
    // Template version for different input sizes
    template<typename ArgsT, size_t N>
    NodeId create_node(const SmallVector<Tensor, N>& inputs, ArgsT&& args) {
        NodeId id = next_id_++;
        
        size_t index = nodes_.size();
        nodes_.emplace_back(id, inputs, std::forward<ArgsT>(args));
        id_to_index_[id] = index;
        
        // Update connectivity for input nodes
        for (const auto& input : inputs) {
            if (!input.is_constant() && input.producer_node() != 0) {
                if (Node* producer = get_node(input.producer_node())) {
                    producer->add_output_node(id);
                }
            }
        }
        
        return id;
    }

    // Find specific operation types
    template<typename ArgsT>
    std::vector<const Node*> find_nodes() const {
        std::vector<const Node*> result;
        OpTypeId target_type = detail::get_op_id<ArgsT>();
        
        for (const auto& node : nodes_) {
            if (node.type_id() == target_type) {
                result.push_back(&node);
            }
        }
        return result;
    }

private:
    std::vector<Node> nodes_;
    std::unordered_map<NodeId, size_t> id_to_index_;
    NodeId next_id_;
};

