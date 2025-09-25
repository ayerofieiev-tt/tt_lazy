#include "core/graph_utils.hpp"
#include "frontend/operations.hpp"
#include "core/common.hpp"

using namespace tt_lazy;

// Function to analyze a computation graph
void analyze_graph(const Tensor& graph, const std::string& graph_name) {
    auto& logger = get_logger();
    
    logger.info("=== Cycle Detection ===");
    bool has_cycles = GraphUtils::has_cycles(graph);
    logger.info("{} graph has cycles: {}", graph_name, has_cycles ? "Yes" : "No");
    
    logger.info("=== Topological Sort - {} Graph ===", graph_name);
    try {
        auto topo_order = GraphUtils::topological_sort(graph);
        logger.info("Topological order:");
        for (size_t i = 0; i < topo_order.size(); ++i) {
            logger.info("  {}. {} [{}]", i + 1, topo_order[i].op_name(), 
                       topo_order[i].shape().to_string());
        }
    } catch (const std::runtime_error& e) {
        logger.error("Error: {}", e.what());
    }

    logger.info("=== DOT Format (for Graphviz) - {} Graph ===", graph_name);
    logger.info("{}", GraphVisualizer::to_dot(graph, graph_name));
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
    // Setup logging
    setup_logging();
    auto& logger = get_logger();
    
    logger.info("=== MLP Graph Visualization Demo ===");
    logger.info("Creating MLP computation graph...");
    
    Shape input_shape({10, 5});
    Shape weight_shape({5, 8});
    Shape bias_shape({8});
    
    logger.debug("Creating tensors with shapes: input={}, weights={}, bias={}", 
                 input_shape.to_string(), weight_shape.to_string(), bias_shape.to_string());
    
    auto mlp_input = zeros(input_shape);
    auto mlp_weights = rand(weight_shape);
    auto mlp_bias = ones(bias_shape);
    auto mlp_final = mlp(mlp_input, mlp_weights, mlp_bias, true);
    
    logger.info("MLP computation graph created successfully");
    analyze_graph(mlp_final, "MLP");
    
    logger.info("Demo completed successfully");
    return 0;
}
