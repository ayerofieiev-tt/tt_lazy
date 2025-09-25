#pragma once
#include "tensor.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tt_lazy {

/**
 * Utility class for graph traversal and analysis operations on Tensor computation graphs.
 * This class provides static methods to analyze and traverse the computation graph
 * without modifying the Tensor class itself.
 */
class GraphUtils {
public:
    // Graph traversal methods
    static std::vector<Tensor> get_all_nodes(const Tensor& root);
    static std::vector<Tensor> get_ancestors(const Tensor& root);
    static std::vector<Tensor> get_descendants(const Tensor& root);
    static std::vector<Tensor> topological_sort(const Tensor& root);
    
    // Graph analysis methods
    static bool has_cycles(const Tensor& root);
    static size_t get_depth(const Tensor& root);
    static size_t get_width(const Tensor& root);
    static size_t get_node_count(const Tensor& root);
    
    // Graph traversal with custom visitor
    template<typename Visitor>
    static void traverse_dfs(const Tensor& root, Visitor&& visitor);
    
    template<typename Visitor>
    static void traverse_bfs(const Tensor& root, Visitor&& visitor);
    
    // Utility methods
    static std::vector<Tensor> get_leaf_nodes(const Tensor& root);
    static std::vector<Tensor> get_root_nodes(const Tensor& root);
    static std::unordered_map<uintptr_t, size_t> get_node_depths(const Tensor& root);
    static std::unordered_map<uintptr_t, size_t> get_node_levels(const Tensor& root);

private:
    // Helper methods for internal use
    static void dfs_visit(const Tensor& node, 
                         std::unordered_set<uintptr_t>& visited,
                         std::function<void(const Tensor&)> visitor);
    
    static void bfs_visit(const Tensor& root,
                         std::function<void(const Tensor&)> visitor);
    
    static bool has_cycles_dfs(const Tensor& node,
                              std::unordered_set<uintptr_t>& visited,
                              std::unordered_set<uintptr_t>& rec_stack);
};

/**
 * Utility class for graph visualization operations.
 * Provides methods to generate various visual representations of the computation graph.
 */
class GraphVisualizer {
public:
    // Visualization formats
    static std::string to_dot(const Tensor& root, const std::string& title = "ComputationGraph");
    static std::string to_ascii_tree(const Tensor& root);
    static std::string to_mermaid(const Tensor& root);
    static std::string to_json(const Tensor& root);
    
    // Print methods
    static void print_graph(const Tensor& root);
    static void print_summary(const Tensor& root);
    static void print_topological_order(const Tensor& root);
    
    // Export methods
    static void export_to_file(const Tensor& root, const std::string& filename, const std::string& format = "dot");
    
    // Customization options
    struct VisualizationOptions {
        bool show_shapes = true;
        bool show_data_types = true;
        bool show_node_ids = false;
        bool color_by_operation = true;
        bool compact_mode = false;
        std::string title = "Computation Graph";
    };
    
    static std::string to_dot(const Tensor& root, const VisualizationOptions& options);
    static std::string to_ascii_tree(const Tensor& root, const VisualizationOptions& options);

private:
    static std::string get_operation_color(const std::string& op_name);
    static std::string format_node_label(const Tensor& node, const VisualizationOptions& options);
};

} // namespace tt_lazy
