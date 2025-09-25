#include "graph_utils.hpp"
#include <iostream>
#include <queue>
#include <sstream>
#include <fstream>

namespace tt_lazy {

// GraphUtils implementation

std::vector<Tensor> GraphUtils::get_all_nodes(const Tensor& root) {
    std::unordered_set<uintptr_t> visited;
    std::vector<Tensor> all_nodes;
    
    dfs_visit(root, visited, [&](const Tensor& node) {
        all_nodes.push_back(node);
    });
    
    return all_nodes;
}

std::vector<Tensor> GraphUtils::get_ancestors(const Tensor& root) {
    std::unordered_set<uintptr_t> visited;
    std::vector<Tensor> ancestors;
    
    dfs_visit(root, visited, [&](const Tensor& node) {
        for (const auto& input : node.inputs()) {
            ancestors.push_back(input);
        }
    });
    
    return ancestors;
}

std::vector<Tensor> GraphUtils::get_descendants(const Tensor& root) {
    std::unordered_set<uintptr_t> visited;
    std::vector<Tensor> descendants;
    
    dfs_visit(root, visited, [&](const Tensor& node) {
        for (const auto& output : node.outputs()) {
            descendants.push_back(output);
        }
    });
    
    return descendants;
}

std::vector<Tensor> GraphUtils::topological_sort(const Tensor& root) {
    std::unordered_set<uintptr_t> visited;
    std::unordered_set<uintptr_t> temp_visited;
    std::vector<Tensor> result;
    
    std::function<void(const Tensor&)> dfs = [&](const Tensor& node) {
        uintptr_t id = node.runtime_id();
        
        if (temp_visited.find(id) != temp_visited.end()) {
            throw std::runtime_error("Cycle detected in computation graph");
        }
        
        if (visited.find(id) != visited.end()) {
            return;
        }
        
        temp_visited.insert(id);
        
        // Visit all input nodes first
        for (const auto& input : node.inputs()) {
            dfs(input);
        }
        
        temp_visited.erase(id);
        visited.insert(id);
        result.push_back(node);
    };
    
    dfs(root);
    return result;
}

bool GraphUtils::has_cycles(const Tensor& root) {
    try {
        topological_sort(root);
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

size_t GraphUtils::get_depth(const Tensor& root) {
    auto depths = get_node_depths(root);
    size_t max_depth = 0;
    for (const auto& pair : depths) {
        max_depth = std::max(max_depth, pair.second);
    }
    return max_depth;
}

size_t GraphUtils::get_width(const Tensor& root) {
    auto levels = get_node_levels(root);
    size_t max_width = 0;
    for (const auto& pair : levels) {
        max_width = std::max(max_width, pair.second);
    }
    return max_width;
}

size_t GraphUtils::get_node_count(const Tensor& root) {
    return get_all_nodes(root).size();
}

template<typename Visitor>
void GraphUtils::traverse_dfs(const Tensor& root, Visitor&& visitor) {
    std::unordered_set<uintptr_t> visited;
    dfs_visit(root, visited, std::forward<Visitor>(visitor));
}

template<typename Visitor>
void GraphUtils::traverse_bfs(const Tensor& root, Visitor&& visitor) {
    bfs_visit(root, std::forward<Visitor>(visitor));
}

std::vector<Tensor> GraphUtils::get_leaf_nodes(const Tensor& root) {
    std::vector<Tensor> leaf_nodes;
    
    traverse_dfs(root, [&](const Tensor& node) {
        if (node.inputs().empty()) {
            leaf_nodes.push_back(node);
        }
    });
    
    return leaf_nodes;
}

std::vector<Tensor> GraphUtils::get_root_nodes(const Tensor& root) {
    std::vector<Tensor> root_nodes;
    
    traverse_dfs(root, [&](const Tensor& node) {
        if (node.outputs().empty()) {
            root_nodes.push_back(node);
        }
    });
    
    return root_nodes;
}

std::unordered_map<uintptr_t, size_t> GraphUtils::get_node_depths(const Tensor& root) {
    std::unordered_map<uintptr_t, size_t> depth_cache;
    
    std::function<size_t(const Tensor&)> dfs = [&](const Tensor& node) -> size_t {
        uintptr_t id = node.runtime_id();
        
        if (depth_cache.find(id) != depth_cache.end()) {
            return depth_cache[id];
        }
        
        size_t max_input_depth = 0;
        for (const auto& input : node.inputs()) {
            max_input_depth = std::max(max_input_depth, dfs(input));
        }
        
        size_t depth = max_input_depth + 1;
        depth_cache[id] = depth;
        return depth;
    };
    
    dfs(root);
    return depth_cache;
}

std::unordered_map<uintptr_t, size_t> GraphUtils::get_node_levels(const Tensor& root) {
    std::unordered_map<uintptr_t, size_t> level_counts;
    
    std::function<void(const Tensor&, size_t)> dfs = [&](const Tensor& node, size_t level) {
        uintptr_t id = node.runtime_id();
        level_counts[level]++;
        
        for (const auto& input : node.inputs()) {
            dfs(input, level + 1);
        }
    };
    
    dfs(root, 0);
    return level_counts;
}

void GraphUtils::dfs_visit(const Tensor& node, 
                          std::unordered_set<uintptr_t>& visited,
                          std::function<void(const Tensor&)> visitor) {
    uintptr_t id = node.runtime_id();
    if (visited.find(id) != visited.end()) {
        return;
    }
    visited.insert(id);
    visitor(node);
    
    // Visit all input nodes
    for (const auto& input : node.inputs()) {
        dfs_visit(input, visited, visitor);
    }
}

void GraphUtils::bfs_visit(const Tensor& root,
                          std::function<void(const Tensor&)> visitor) {
    std::queue<Tensor> queue;
    std::unordered_set<uintptr_t> visited;
    
    queue.push(root);
    visited.insert(root.runtime_id());
    
    while (!queue.empty()) {
        Tensor current = queue.front();
        queue.pop();
        
        visitor(current);
        
        // Add all inputs to queue
        for (const auto& input : current.inputs()) {
            uintptr_t id = input.runtime_id();
            if (visited.find(id) == visited.end()) {
                visited.insert(id);
                queue.push(input);
            }
        }
    }
}

// GraphVisualizer implementation

std::string GraphVisualizer::to_dot(const Tensor& root, const std::string& title) {
    return to_dot(root, VisualizationOptions{.title = title});
}

std::string GraphVisualizer::to_dot(const Tensor& root, const VisualizationOptions& options) {
    std::ostringstream oss;
    std::unordered_set<uintptr_t> visited;
    
    oss << "digraph " << options.title << " {\n";
    oss << "  rankdir=TB;\n";
    oss << "  node [shape=box, style=filled];\n";
    oss << "  edge [color=gray];\n\n";
    
    std::function<void(const Tensor&)> dfs = [&](const Tensor& node) {
        uintptr_t id = node.runtime_id();
        if (visited.find(id) != visited.end()) {
            return;
        }
        visited.insert(id);
        
        // Create node
        std::string node_name = "node_" + std::to_string(id);
        std::string label = format_node_label(node, options);
        std::string color = options.color_by_operation ? get_operation_color(std::string(node.op_name())) : "lightblue";
        
        oss << "  " << node_name << " [label=\"" << label << "\", fillcolor=\"" << color << "\"];\n";
        
        // Create edges to inputs
        for (const auto& input : node.inputs()) {
            std::string input_name = "node_" + std::to_string(input.runtime_id());
            oss << "  " << input_name << " -> " << node_name << ";\n";
            dfs(input);
        }
    };
    
    dfs(root);
    oss << "}\n";
    return oss.str();
}

std::string GraphVisualizer::to_ascii_tree(const Tensor& root) {
    return to_ascii_tree(root, VisualizationOptions{});
}

std::string GraphVisualizer::to_ascii_tree(const Tensor& root, const VisualizationOptions& options) {
    std::ostringstream oss;
    std::unordered_set<uintptr_t> visited;
    
    std::function<void(const Tensor&, const std::string&, bool)> dfs = 
        [&](const Tensor& node, const std::string& prefix, bool is_last) {
        
        uintptr_t id = node.runtime_id();
        if (visited.find(id) != visited.end()) {
            oss << prefix << (is_last ? "└── " : "├── ") 
                << "[CYCLE: " << node.op_name() << "]\n";
            return;
        }
        visited.insert(id);
        
        std::string label = std::string(node.op_name());
        if (options.show_shapes) {
            label += " [" + node.shape().to_string() + "]";
        }
        if (options.show_node_ids) {
            label += " (id:" + std::to_string(id) + ")";
        }
        
        oss << prefix << (is_last ? "└── " : "├── ") << label << "\n";
        
        const auto& inputs = node.inputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            bool is_last_input = (i == inputs.size() - 1);
            std::string new_prefix = prefix + (is_last ? "    " : "│   ");
            dfs(inputs[i], new_prefix, is_last_input);
        }
    };
    
    dfs(root, "", true);
    return oss.str();
}

std::string GraphVisualizer::to_mermaid(const Tensor& root) {
    std::ostringstream oss;
    std::unordered_set<uintptr_t> visited;
    
    oss << "graph TD\n";
    
    std::function<void(const Tensor&)> dfs = [&](const Tensor& node) {
        uintptr_t id = node.runtime_id();
        if (visited.find(id) != visited.end()) {
            return;
        }
        visited.insert(id);
        
        std::string node_id = "N" + std::to_string(id);
        std::string label = std::string(node.op_name()) + "\\n" + node.shape().to_string();
        
        oss << "  " << node_id << "[\"" << label << "\"]\n";
        
        for (const auto& input : node.inputs()) {
            std::string input_id = "N" + std::to_string(input.runtime_id());
            oss << "  " << input_id << " --> " << node_id << "\n";
            dfs(input);
        }
    };
    
    dfs(root);
    return oss.str();
}

std::string GraphVisualizer::to_json(const Tensor& root) {
    std::ostringstream oss;
    std::unordered_set<uintptr_t> visited;
    
    oss << "{\n";
    oss << "  \"nodes\": [\n";
    
    std::vector<std::string> node_entries;
    std::vector<std::string> edge_entries;
    
    std::function<void(const Tensor&)> dfs = [&](const Tensor& node) {
        uintptr_t id = node.runtime_id();
        if (visited.find(id) != visited.end()) {
            return;
        }
        visited.insert(id);
        
        std::ostringstream node_oss;
        node_oss << "    {\n";
        node_oss << "      \"id\": " << id << ",\n";
        node_oss << "      \"operation\": \"" << node.op_name() << "\",\n";
        node_oss << "      \"shape\": \"" << node.shape().to_string() << "\",\n";
        node_oss << "      \"dtype\": " << static_cast<int>(node.dtype()) << "\n";
        node_oss << "    }";
        node_entries.push_back(node_oss.str());
        
        for (const auto& input : node.inputs()) {
            std::ostringstream edge_oss;
            edge_oss << "    {\n";
            edge_oss << "      \"from\": " << input.runtime_id() << ",\n";
            edge_oss << "      \"to\": " << id << "\n";
            edge_oss << "    }";
            edge_entries.push_back(edge_oss.str());
            dfs(input);
        }
    };
    
    dfs(root);
    
    // Output nodes
    for (size_t i = 0; i < node_entries.size(); ++i) {
        oss << node_entries[i];
        if (i < node_entries.size() - 1) oss << ",";
        oss << "\n";
    }
    
    oss << "  ],\n";
    oss << "  \"edges\": [\n";
    
    // Output edges
    for (size_t i = 0; i < edge_entries.size(); ++i) {
        oss << edge_entries[i];
        if (i < edge_entries.size() - 1) oss << ",";
        oss << "\n";
    }
    
    oss << "  ]\n";
    oss << "}\n";
    
    return oss.str();
}

void GraphVisualizer::print_graph(const Tensor& root) {
    print_summary(root);
    std::cout << "\n=== ASCII Tree ===\n";
    std::cout << to_ascii_tree(root) << "\n";
    std::cout << "\n=== DOT Format ===\n";
    std::cout << to_dot(root) << "\n";
}

void GraphVisualizer::print_summary(const Tensor& root) {
    std::cout << "=== Computation Graph Summary ===\n";
    std::cout << "Depth: " << GraphUtils::get_depth(root) << "\n";
    std::cout << "Width: " << GraphUtils::get_width(root) << "\n";
    std::cout << "Has cycles: " << (GraphUtils::has_cycles(root) ? "Yes" : "No") << "\n";
    std::cout << "Total nodes: " << GraphUtils::get_node_count(root) << "\n";
    std::cout << "Leaf nodes: " << GraphUtils::get_leaf_nodes(root).size() << "\n";
    std::cout << "Root nodes: " << GraphUtils::get_root_nodes(root).size() << "\n";
}

void GraphVisualizer::print_topological_order(const Tensor& root) {
    try {
        auto topo_order = GraphUtils::topological_sort(root);
        std::cout << "=== Topological Order ===\n";
        for (size_t i = 0; i < topo_order.size(); ++i) {
            std::cout << i + 1 << ". " << topo_order[i].op_name() 
                      << " [" << topo_order[i].shape().to_string() << "]\n";
        }
    } catch (const std::runtime_error& e) {
        std::cout << "Cannot compute topological order: " << e.what() << "\n";
    }
}

void GraphVisualizer::export_to_file(const Tensor& root, const std::string& filename, const std::string& format) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    if (format == "dot") {
        file << to_dot(root);
    } else if (format == "mermaid") {
        file << to_mermaid(root);
    } else if (format == "json") {
        file << to_json(root);
    } else {
        throw std::runtime_error("Unsupported format: " + format);
    }
    
    file.close();
}

std::string GraphVisualizer::get_operation_color(const std::string& op_name) {
    // Simple color mapping based on operation type
    if (op_name.find("MatMul") != std::string::npos) return "lightcoral";
    if (op_name.find("Add") != std::string::npos) return "lightgreen";
    if (op_name.find("Multiply") != std::string::npos) return "lightyellow";
    if (op_name.find("ReLU") != std::string::npos) return "lightpink";
    if (op_name.find("Reduce") != std::string::npos) return "lightcyan";
    if (op_name.find("Split") != std::string::npos) return "lightgray";
    if (op_name.find("FusedMLP") != std::string::npos) return "lightsteelblue";
    return "lightblue";
}

std::string GraphVisualizer::format_node_label(const Tensor& node, const VisualizationOptions& options) {
    std::ostringstream oss;
    oss << node.op_name();
    
    if (options.show_shapes) {
        oss << "\\nShape: " << node.shape().to_string();
    }
    
    if (options.show_data_types) {
        oss << "\\nType: " << static_cast<int>(node.dtype());
    }
    
    if (options.show_node_ids) {
        oss << "\\nID: " << node.runtime_id();
    }
    
    return oss.str();
}

} // namespace tt_lazy
