#include "core/graph_utils.hpp"
#include "frontend/operations.hpp"

#include <iostream>

using namespace tt_lazy;

// Function to analyze a computation graph
void analyze_graph(const Tensor& graph, const std::string& graph_name) {
    std::cout << "=== Cycle Detection ===\n";
    bool has_cycles = GraphUtils::has_cycles(graph);
    std::cout << graph_name << " graph has cycles: " << (has_cycles ? "Yes" : "No") << "\n\n";
    
    std::cout << "=== Topological Sort - " << graph_name << " Graph ===\n";
    try {
        auto topo_order = GraphUtils::topological_sort(graph);
        std::cout << "Topological order:\n";
        for (size_t i = 0; i < topo_order.size(); ++i) {
            std::cout << "  " << i + 1 << ". " << topo_order[i].op_name() 
                      << " [" << topo_order[i].shape().to_string() << "]\n";
        }
    } catch (const std::runtime_error& e) {
        std::cout << "Error: " << e.what() << "\n";
    }
    std::cout << "\n";

    std::cout << "=== DOT Format (for Graphviz) - " << graph_name << " Graph ===\n";
    std::cout << GraphVisualizer::to_dot(graph, graph_name) << "\n";
}

// MLP function that creates a multi-layer perceptron computation graph
Tensor mlp(const Tensor& input, const Tensor& weights, const Tensor& bias, 
           bool apply_reduce_sum = true) {
    // MLP computation: ReLU(input @ weights + bias)
    auto mlp_matmul = matmul(input, weights);
    auto mlp_add_bias = add(mlp_matmul, bias);
    auto mlp_output = relu(mlp_add_bias);
    
    if (apply_reduce_sum) {
        // Reduce along the last dimension, keep dimensions
        return reduce_sum(mlp_output, /*dims*/std::vector<int32_t>{1}, /*keepdims*/true);
    }
    
    return mlp_output;
}

int main() {
    std::cout << "=== MLP Graph Visualization Demo ===\n\n";
    std::cout << "Creating MLP computation graph...\n";
    
    Shape input_shape({10, 5});
    Shape weight_shape({5, 8});
    Shape bias_shape({8});
    
    auto mlp_input = zeros(input_shape);
    auto mlp_weights = rand(weight_shape);
    auto mlp_bias = ones(bias_shape);
    auto mlp_final = mlp(mlp_input, mlp_weights, mlp_bias, true);
    
    analyze_graph(mlp_final, "MLP");
    return 0;
}
