#pragma once
#include "common.hpp"
#include "OpArgs.hpp"
#include <string>
#include <ostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>

// Forward declarations
class Context;
class Node;

// Unified Tensor class that can be either lazy (graph-based) or materialized (data-based)
class Tensor {
public:
    // Tensor state
    enum class State {
        LAZY,        // Contains graph node information
        MATERIALIZED // Contains actual data
    };

    // Default constructor - null tensor
    Tensor();
    
    // Create lazy tensor from node output
    Tensor(NodeId producer, uint16_t output_idx, std::initializer_list<uint32_t> shape);
    
    // Create materialized tensor with data
    Tensor(std::initializer_list<uint32_t> shape);
    Tensor(const std::vector<uint32_t>& shape);
    Tensor(const std::vector<uint32_t>& shape, const std::vector<float>& data);
    Tensor(void* data, std::initializer_list<uint32_t> shape); // For constants
    
    // Copy/move constructors
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor();
    
    // State information
    State state() const { return state_; }
    bool is_lazy() const { return state_ == State::LAZY; }
    bool is_materialized() const { return state_ == State::MATERIALIZED; }
    bool is_constant() const { return is_constant_; }
    bool is_null() const;
    explicit operator bool() const;
    
    // Lazy tensor methods
    NodeId producer_node() const;
    uint16_t output_index() const;
    
    // Shape information (works for both states)
    const uint32_t* shape() const;
    uint16_t rank() const;
    uint32_t size(size_t dim) const;
    size_t total_elements() const;
    bool is_scalar() const;
    
    // Data access (requires materialization for lazy tensors)
    float* data_ptr();
    const float* const_data_ptr() const;
    std::vector<float> to_vector() const;
    
    // Materialization methods
    void materialize();
    void force_materialization();
    
    // Graph visualization methods (for lazy tensors)
    std::string to_string() const;
    void print_graph(std::ostream& os = std::cout, int indent = 0) const;
    
    // Helper for graph traversal
    struct GraphNode {
        NodeId id;
        std::string op_name;
        std::vector<std::string> args;
        std::vector<GraphNode> inputs;
        int depth;
    };
    
    GraphNode build_graph_node(int max_depth = 10) const;
    
    // Utility methods
    void fill(float value);
    void print() const;
    Tensor reshape(const std::vector<uint32_t>& new_shape) const;
    
    // Broadcasting helpers
    static std::vector<uint32_t> broadcast_shapes(const std::vector<uint32_t>& shape1, 
                                                 const std::vector<uint32_t>& shape2);
    static bool can_broadcast(const std::vector<uint32_t>& shape1, 
                             const std::vector<uint32_t>& shape2);

private:
    State state_;
    
    // Lazy state data
    NodeId producer_node_;
    uint16_t output_index_;
    
    // Shape information (common to both states)
    uint16_t rank_;
    uint32_t shape_[4];
    
    // Materialized state data
    std::unique_ptr<float[]> data_;
    size_t numel_;
    
    // Constant flag
    bool is_constant_;
    void* constant_data_; // For constants only
    
    // Materialization cache
    mutable std::shared_ptr<Tensor> materialized_cache_;
    mutable std::atomic<bool> materialization_in_progress_;
    
    // Helper methods
    void allocate_data();
    size_t compute_numel() const;
    void materialize_impl() const;
    void copy_from_other(const Tensor& other);
    void move_from_other(Tensor&& other);
};

// Stream operator for easy printing
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);