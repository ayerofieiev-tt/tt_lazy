# Graph Visualization and Traversal

This document describes how to use the graph visualization and traversal utilities in the tt_lazy library.

## Overview

The graph utilities are provided through two main classes:
- `GraphUtils`: Static methods for graph analysis and traversal
- `GraphVisualizer`: Static methods for graph visualization and export

## Creating Constant Tensors

Before creating computation graphs, you need to create input/constant tensors. The library provides several convenient functions:

```cpp
#include "frontend/operations.hpp"

// Create constant tensors with different shapes and data types
auto tensor1 = constant({2, 3}, DataType::FLOAT32);           // Using vector shape
auto tensor2 = constant(Shape({2, 3}), DataType::FLOAT32);    // Using Shape object
auto tensor3 = zeros({2, 3}, DataType::FLOAT32);              // Create zeros tensor
auto tensor4 = ones({2, 3}, DataType::FLOAT32);               // Create ones tensor

// Default data type is FLOAT32
auto tensor5 = constant({2, 3});                              // Defaults to FLOAT32
auto tensor6 = zeros({2, 3});                                 // Defaults to FLOAT32
auto tensor7 = ones({2, 3});                                  // Defaults to FLOAT32
```

## GraphUtils - Graph Analysis

### Basic Traversal

```cpp
#include "core/graph_utils.hpp"

// Get all nodes in the computation graph
auto all_nodes = GraphUtils::get_all_nodes(tensor);

// Get all ancestor nodes (inputs)
auto ancestors = GraphUtils::get_ancestors(tensor);

// Get all descendant nodes (outputs)
auto descendants = GraphUtils::get_descendants(tensor);
```

### Graph Analysis

```cpp
// Check for cycles in the graph
bool has_cycles = GraphUtils::has_cycles(tensor);

// Get graph depth (longest path from root to leaf)
size_t depth = GraphUtils::get_depth(tensor);

// Get graph width (maximum number of nodes at any level)
size_t width = GraphUtils::get_width(tensor);

// Get total number of nodes
size_t node_count = GraphUtils::get_node_count(tensor);
```

### Topological Sort

```cpp
try {
    auto topo_order = GraphUtils::topological_sort(tensor);
    // Process nodes in topological order
    for (const auto& node : topo_order) {
        // Process node...
    }
} catch (const std::runtime_error& e) {
    // Handle cycle detection
    std::cout << "Cycle detected: " << e.what() << std::endl;
}
```

### Custom Traversal

```cpp
// Depth-first traversal with custom visitor
GraphUtils::traverse_dfs(tensor, [](const Tensor& node) {
    std::cout << "Visiting: " << node.op_name() << std::endl;
});

// Breadth-first traversal with custom visitor
GraphUtils::traverse_bfs(tensor, [](const Tensor& node) {
    std::cout << "Visiting: " << node.op_name() << std::endl;
});
```

### Utility Methods

```cpp
// Get leaf nodes (nodes with no inputs)
auto leaf_nodes = GraphUtils::get_leaf_nodes(tensor);

// Get root nodes (nodes with no outputs)
auto root_nodes = GraphUtils::get_root_nodes(tensor);

// Get depth of each node
auto node_depths = GraphUtils::get_node_depths(tensor);

// Get level of each node (distance from root)
auto node_levels = GraphUtils::get_node_levels(tensor);
```

## GraphVisualizer - Graph Visualization

### Basic Visualization

```cpp
#include "core/graph_utils.hpp"

// Print comprehensive graph information
GraphVisualizer::print_graph(tensor);

// Print just the summary
GraphVisualizer::print_summary(tensor);

// Print topological order
GraphVisualizer::print_topological_order(tensor);
```

### ASCII Tree Visualization

```cpp
// Generate ASCII tree representation
std::string ascii_tree = GraphVisualizer::to_ascii_tree(tensor);
std::cout << ascii_tree << std::endl;
```

### DOT Format (Graphviz)

```cpp
// Generate DOT format for Graphviz
std::string dot_format = GraphVisualizer::to_dot(tensor, "MyGraph");
std::cout << dot_format << std::endl;

// Save to file and render with Graphviz
GraphVisualizer::export_to_file(tensor, "graph.dot", "dot");
// Then run: dot -Tpng graph.dot -o graph.png
```

### Mermaid Format

```cpp
// Generate Mermaid format
std::string mermaid = GraphVisualizer::to_mermaid(tensor);
std::cout << mermaid << std::endl;

// Export to file
GraphVisualizer::export_to_file(tensor, "graph.mmd", "mermaid");
```

### JSON Format

```cpp
// Generate JSON representation
std::string json = GraphVisualizer::to_json(tensor);
std::cout << json << std::endl;

// Export to file
GraphVisualizer::export_to_file(tensor, "graph.json", "json");
```

### Custom Visualization Options

```cpp
GraphVisualizer::VisualizationOptions options;
options.show_shapes = true;           // Show tensor shapes
options.show_data_types = true;       // Show data types
options.show_node_ids = false;        // Show node IDs
options.color_by_operation = true;    // Color nodes by operation type
options.compact_mode = false;         // Use compact representation
options.title = "My Custom Graph";    // Custom title

// Generate DOT with custom options
std::string custom_dot = GraphVisualizer::to_dot(tensor, options);

// Generate ASCII tree with custom options
std::string custom_ascii = GraphVisualizer::to_ascii_tree(tensor, options);
```

## Example Usage

Here's a complete example showing how to create and visualize a computation graph:

```cpp
#include "core/graph_utils.hpp"
#include "frontend/operations.hpp"

int main() {
    // Create a simple computation graph: (a + b) * c
    Shape shape({2, 3});
    DataType dtype = DataType::FLOAT32;
    
    // Create input tensors using constant creation functions
    auto input_a = constant(shape, dtype);
    auto input_b = zeros(shape, dtype);        // Alternative: zeros()
    auto input_c = ones(shape, dtype);         // Alternative: ones()
    
    // Create intermediate result: a + b
    auto add_result = add(input_a, input_b);
    
    // Create final result: (a + b) * c
    auto final_result = multiply(add_result, input_c);
    
    // Analyze the graph
    std::cout << "Graph depth: " << GraphUtils::get_depth(final_result) << std::endl;
    std::cout << "Graph width: " << GraphUtils::get_width(final_result) << std::endl;
    std::cout << "Has cycles: " << (GraphUtils::has_cycles(final_result) ? "Yes" : "No") << std::endl;
    
    // Visualize the graph
    GraphVisualizer::print_graph(final_result);
    
    // Export to various formats
    GraphVisualizer::export_to_file(final_result, "graph.dot", "dot");
    GraphVisualizer::export_to_file(final_result, "graph.json", "json");
    GraphVisualizer::export_to_file(final_result, "graph.mmd", "mermaid");
    
    return 0;
}
```

## Integration with External Tools

### Graphviz
1. Export graph to DOT format: `GraphVisualizer::export_to_file(tensor, "graph.dot", "dot")`
2. Render with Graphviz: `dot -Tpng graph.dot -o graph.png`
3. Or render as SVG: `dot -Tsvg graph.dot -o graph.svg`

### Mermaid
1. Export graph to Mermaid format: `GraphVisualizer::export_to_file(tensor, "graph.mmd", "mermaid")`
2. Use in Mermaid Live Editor or GitHub markdown

### Custom Tools
1. Export to JSON format: `GraphVisualizer::export_to_file(tensor, "graph.json", "json")`
2. Parse JSON in your preferred language/tool

## Performance Considerations

- Graph traversal methods use DFS/BFS algorithms with O(V + E) complexity
- Cycle detection uses DFS with temporary marking, also O(V + E)
- Visualization methods may be expensive for large graphs due to string formatting
- Consider using custom visitors for large graphs instead of collecting all nodes

## Thread Safety

- All methods are thread-safe for read-only operations
- Multiple threads can safely call these methods on different tensors
- Concurrent access to the same tensor is safe as long as the tensor is not being modified
