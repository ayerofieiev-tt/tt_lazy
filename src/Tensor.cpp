#include "Tensor.hpp"

#include "Context.hpp"
#include "EvaluationManager.hpp"
#include "Node.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

#include <spdlog/spdlog.h>

// Default constructor - null tensor
Tensor::Tensor()
    : state_(State::LAZY),
      producer_node_(0),
      output_index_(0),
      rank_(0),
      shape_{},
      data_(nullptr),
      numel_(0),
      is_constant_(false),
      constant_data_(nullptr),
      evaluation_in_progress_(false) {
}

// Create lazy tensor from node output
Tensor::Tensor(
    NodeId producer_node_id, uint16_t output_index,
    std::initializer_list<uint32_t>
        shape)  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init,bugprone-easily-swappable-parameters) - shape_ initialized in body, parameters semantically different
    : state_(State::LAZY),
      producer_node_(producer_node_id),
      output_index_(output_index),
      rank_(static_cast<uint16_t>(shape.size())),
      data_(nullptr),
      numel_(0),
      is_constant_(false),
      constant_data_(nullptr),
      evaluation_in_progress_(false) {
    assert(rank_ <= 4);
    std::copy(
        shape.begin(), shape.end(),
        shape_);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    std::fill(shape_ + rank_, shape_ + 4,
              1);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    numel_ = compute_numel();
}

// Create materialized tensor with shape only
Tensor::Tensor(
    std::initializer_list<uint32_t>
        shape)  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) - shape_ initialized in body
    : state_(State::MATERIALIZED),
      producer_node_(0),
      output_index_(0),
      rank_(static_cast<uint16_t>(shape.size())),
      is_constant_(false),
      constant_data_(nullptr),
      evaluation_in_progress_(false) {
    assert(rank_ <= 4);
    std::copy(
        shape.begin(), shape.end(),
        shape_);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    std::fill(shape_ + rank_, shape_ + 4,
              1);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    numel_ = compute_numel();
    allocate_data();
}

Tensor::Tensor(
    const std::vector<uint32_t>&
        shape)  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) - shape_ initialized in body
    : state_(State::MATERIALIZED),
      producer_node_(0),
      output_index_(0),
      rank_(static_cast<uint16_t>(shape.size())),
      is_constant_(false),
      constant_data_(nullptr),
      evaluation_in_progress_(false) {
    assert(rank_ <= 4);
    std::copy(
        shape.begin(), shape.end(),
        shape_);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    std::fill(shape_ + rank_, shape_ + 4,
              1);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    numel_ = compute_numel();
    allocate_data();
}

Tensor::Tensor(
    const std::vector<uint32_t>& shape,
    const std::vector<float>&
        data)  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) - shape_ initialized in body
    : state_(State::MATERIALIZED),
      producer_node_(0),
      output_index_(0),
      rank_(static_cast<uint16_t>(shape.size())),
      is_constant_(false),
      constant_data_(nullptr),
      evaluation_in_progress_(false) {
    assert(rank_ <= 4);
    std::copy(
        shape.begin(), shape.end(),
        shape_);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    std::fill(shape_ + rank_, shape_ + 4,
              1);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    numel_ = compute_numel();
    allocate_data();

    // Copy data
    std::copy(data.begin(), data.end(), data_.get());
}

// Create constant tensor
Tensor::Tensor(
    void* data,
    std::initializer_list<uint32_t>
        shape)  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) - shape_ initialized in body
    : state_(State::MATERIALIZED),
      producer_node_(0),
      output_index_(0),
      rank_(static_cast<uint16_t>(shape.size())),
      data_(nullptr),
      numel_(0),
      is_constant_(true),
      constant_data_(data),
      evaluation_in_progress_(false) {
    assert(rank_ <= 4);
    std::copy(
        shape.begin(), shape.end(),
        shape_);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    std::fill(shape_ + rank_, shape_ + 4,
              1);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay) - Safe array access with known bounds
    numel_ = compute_numel();
}

// Copy constructor
Tensor::Tensor(
    const Tensor&
        other)  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) - shape_ initialized in body
    : state_(other.state_),
      producer_node_(other.producer_node_),
      output_index_(other.output_index_),
      rank_(other.rank_),
      numel_(other.numel_),
      is_constant_(other.is_constant_),
      constant_data_(other.constant_data_),
      evaluation_in_progress_(false) {
    std::copy(other.shape_, other.shape_ + 4, shape_);
    copy_from_other(other);
}

// Move constructor
Tensor::Tensor(
    Tensor&&
        other) noexcept  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) - shape_ initialized in body
    : state_(other.state_),
      producer_node_(other.producer_node_),
      output_index_(other.output_index_),
      rank_(other.rank_),
      numel_(other.numel_),
      is_constant_(other.is_constant_),
      constant_data_(other.constant_data_),
      evaluation_in_progress_(false) {
    std::copy(other.shape_, other.shape_ + 4, shape_);
    move_from_other(std::move(other));
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        state_ = other.state_;
        producer_node_ = other.producer_node_;
        output_index_ = other.output_index_;
        rank_ = other.rank_;
        numel_ = other.numel_;
        is_constant_ = other.is_constant_;
        constant_data_ = other.constant_data_;
        std::copy(other.shape_, other.shape_ + 4, shape_);
        copy_from_other(other);
    }
    return *this;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        state_ = other.state_;
        producer_node_ = other.producer_node_;
        output_index_ = other.output_index_;
        rank_ = other.rank_;
        numel_ = other.numel_;
        is_constant_ = other.is_constant_;
        constant_data_ = other.constant_data_;
        std::copy(other.shape_, other.shape_ + 4, shape_);
        move_from_other(std::move(other));
    }
    return *this;
}

// Destructor
Tensor::~Tensor() = default;

// State information
bool Tensor::is_null() const {
    return state_ == State::LAZY && producer_node_ == 0;
}

Tensor::operator bool() const {
    return !is_null();
}

// Lazy tensor methods
NodeId Tensor::producer_node() const {
    return producer_node_;
}

uint16_t Tensor::output_index() const {
    return output_index_;
}

// Shape information
const uint32_t* Tensor::shape() const {
    return shape_;
}

uint16_t Tensor::rank() const {
    return rank_;
}

uint32_t Tensor::size(size_t dim) const {
    return dim < rank_
               ? shape_[dim]
               : 1;  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
}

size_t Tensor::total_elements() const {
    return numel_;
}

bool Tensor::is_scalar() const {
    return total_elements() == 1;
}

// Data access
float* Tensor::data_ptr() {
    if (state_ == State::LAZY) {
        eval();
    }

    if (is_constant_) {
        return static_cast<float*>(constant_data_);
    }

    return data_.get();
}

const float* Tensor::const_data_ptr() const {
    if (state_ == State::LAZY) {
        const_cast<Tensor*>(this)
            ->eval();  // NOLINT(cppcoreguidelines-pro-type-const-cast) - Lazy evaluation requires mutable access
    }

    if (is_constant_) {
        return static_cast<const float*>(constant_data_);
    }

    return data_.get();
}

std::vector<float> Tensor::to_vector() const {
    const float* data = const_data_ptr();
    if (!data) {
        return {};
    }

    std::vector<float> vec(numel_);
    std::copy(data, data + numel_, vec.begin());
    return vec;
}

// Evaluation methods
void Tensor::eval() {
    if (state_ == State::MATERIALIZED) {
        return;
    }

    eval_impl();
}

// Graph visualization methods
Tensor::GraphNode Tensor::build_graph_node(int max_depth) const {
    GraphNode node;
    node.depth = 0;

    if (is_constant()) {
        node.id = 0;
        node.op_name = "CONSTANT";
        std::ostringstream shape_str;
        shape_str << "shape=[";
        for (int i = 0; i < rank_; ++i) {
            if (i > 0)
                shape_str << ", ";
            shape_str << shape_
                    [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
        }
        shape_str << "]";
        node.args.push_back(shape_str.str());
        return node;
    }

    if (is_null()) {
        node.id = 0;
        node.op_name = "NULL";
        return node;
    }

    if (state_ == State::MATERIALIZED) {
        node.id = 0;
        node.op_name = "MATERIALIZED";
        std::ostringstream shape_str;
        shape_str << "shape=[";
        for (int i = 0; i < rank_; ++i) {
            if (i > 0)
                shape_str << ", ";
            shape_str << shape_
                    [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
        }
        shape_str << "]";
        node.args.push_back(shape_str.str());
        return node;
    }

    const Node* producer = Context::instance().get_node(producer_node_);
    if (!producer) {
        node.id = producer_node_;
        node.op_name = "UNKNOWN";
        return node;
    }

    node.id = producer_node_;
    node.op_name = std::string(producer->op_name());

    // Add operation type ID for generic identification
    node.args.push_back("type_id=" + std::to_string(producer->type_id()));

    // Add shape information
    std::ostringstream shape_str;
    shape_str << "shape=[";
    for (int i = 0; i < rank_; ++i) {
        if (i > 0)
            shape_str << ", ";
        shape_str << shape_
                [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
    }
    shape_str << "]";
    node.args.push_back(shape_str.str());

    // Recursively build input nodes (with depth limit to prevent infinite recursion)
    if (max_depth > 0) {
        for (const auto& input : producer->inputs()) {
            if (!input.is_constant() && !input.is_null()) {
                Tensor::GraphNode input_node = input.build_graph_node(max_depth - 1);
                input_node.depth = node.depth + 1;
                node.inputs.push_back(input_node);
            }
        }
    }

    return node;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    print_graph(oss, 0);
    return oss.str();
}

void Tensor::print_graph(std::ostream& os, int indent) const {
    GraphNode root = build_graph_node();

    std::function<void(const GraphNode&, int)> print_node = [&](const GraphNode& node, int current_indent) {
        // Build indentation string
        std::string indent_str(static_cast<size_t>(current_indent * 2), ' ');

        // Build operation string
        std::string op_str = "[" + std::to_string(node.id) + "] " + node.op_name;

        // Add arguments if any
        if (!node.args.empty()) {
            op_str += "(";
            for (size_t i = 0; i < node.args.size(); ++i) {
                if (i > 0)
                    op_str += ", ";
                op_str += node.args[i];
            }
            op_str += ")";
        }

        // Output the operation
        os << indent_str << op_str << "\n";

        // Print inputs recursively
        for (const auto& input : node.inputs) {
            print_node(input, current_indent + 1);
        }
    };

    print_node(root, indent);
}

void Tensor::print_graph(int indent) const {
    std::ostringstream oss;
    print_graph(oss, indent);
    spdlog::info("{}", oss.str());
}

// Utility methods
void Tensor::fill(float value) {
    if (state_ == State::LAZY) {
        eval();
    }

    float* data = data_ptr();
    if (data) {
        std::fill(data, data + numel_, value);
    }
}

void Tensor::print() const {
    if (state_ == State::LAZY) {
        print_graph();
        return;
    }

    const float* data = const_data_ptr();
    if (!data) {
        spdlog::info("Empty tensor");
        return;
    }

    // Build shape string
    std::ostringstream shape_stream;
    shape_stream << "Tensor shape: [";
    for (int i = 0; i < rank_; ++i) {
        if (i > 0)
            shape_stream << ", ";
        shape_stream << shape_
                [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
    }
    shape_stream << "]";
    spdlog::info(shape_stream.str());

    // Print data (simplified for small tensors)
    static constexpr size_t MAX_PRINT_ELEMENTS = 16;
    if (numel_ <= MAX_PRINT_ELEMENTS) {
        std::ostringstream data_stream;
        data_stream << "Data: [";
        for (size_t i = 0; i < numel_; ++i) {
            if (i > 0)
                data_stream << ", ";
            data_stream << data[i];
        }
        data_stream << "]";
        spdlog::info(data_stream.str());
    } else {
        spdlog::info("Data: [too large to display]");
    }
}

Tensor Tensor::reshape(const std::vector<uint32_t>& new_shape) const {
    // Verify total elements match
    size_t new_numel = 1;
    for (uint32_t dim : new_shape) {
        new_numel *= dim;
    }

    if (new_numel != numel_) {
        throw std::runtime_error("Reshape: total elements mismatch");
    }

    Tensor result = *this;
    result.rank_ = static_cast<uint16_t>(new_shape.size());
    assert(result.rank_ <= 4);
    std::copy(new_shape.begin(), new_shape.end(), result.shape_);
    std::fill(result.shape_ + result.rank_, result.shape_ + 4, 1);

    return result;
}

// Broadcasting helpers
std::vector<uint32_t> Tensor::broadcast_shapes(const std::vector<uint32_t>& shape1,
                                               const std::vector<uint32_t>& shape2) {
    size_t max_rank = std::max(shape1.size(), shape2.size());
    std::vector<uint32_t> result(max_rank);

    for (size_t i = 0; i < max_rank; ++i) {
        size_t idx1 = shape1.size() - 1 - i;
        size_t idx2 = shape2.size() - 1 - i;

        uint32_t dim1 = (idx1 < shape1.size()) ? shape1[idx1] : 1;
        uint32_t dim2 = (idx2 < shape2.size()) ? shape2[idx2] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::runtime_error("Incompatible shapes for broadcasting");
        }

        result[max_rank - 1 - i] = std::max(dim1, dim2);
    }

    return result;
}

bool Tensor::can_broadcast(const std::vector<uint32_t>& shape1, const std::vector<uint32_t>& shape2) {
    try {
        broadcast_shapes(shape1, shape2);
        return true;
    } catch (...) {
        return false;
    }
}

// Helper methods
void Tensor::allocate_data() {
    if (numel_ > 0) {
        data_ = std::make_unique<float[]>(
            numel_);  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Dynamic array for tensor data
    }
}

size_t Tensor::compute_numel() const {
    size_t total = 1;
    for (size_t i = 0; i < rank_; ++i) {
        total *= shape_
            [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
    }
    return total;
}

void Tensor::eval_impl() const {
    if (state_ == State::MATERIALIZED) {
        return;
    }

    // Check if evaluation is already in progress (prevent infinite recursion)
    if (evaluation_in_progress_.load()) {
        return;
    }

    evaluation_in_progress_.store(true);

    try {
        // Use the evaluation manager to evaluate this tensor
        auto& eval_manager = tt_lazy::get_evaluation_manager();
        auto evaluated = eval_manager.evaluate(*this);

        if (evaluated) {
            // Copy the evaluated data to this tensor
            const_cast<Tensor*>(this)->state_ = State::
                MATERIALIZED;  // NOLINT(cppcoreguidelines-pro-type-const-cast) - Lazy evaluation requires mutable access
            const_cast<Tensor*>(this)
                ->allocate_data();  // NOLINT(cppcoreguidelines-pro-type-const-cast) - Lazy evaluation requires mutable access

            const float* src_data = evaluated->const_data_ptr();
            float* dst_data =
                const_cast<Tensor*>(this)
                    ->data_ptr();  // NOLINT(cppcoreguidelines-pro-type-const-cast) - Lazy evaluation requires mutable access
            if (src_data && dst_data) {
                std::memcpy(dst_data, src_data, numel_ * sizeof(float));
            }
        } else {
            throw std::runtime_error("Failed to evaluate tensor");
        }
    } catch (...) {
        // Handle evaluation failure
        evaluation_in_progress_.store(false);
        throw;
    }

    evaluation_in_progress_.store(false);
}

void Tensor::copy_from_other(const Tensor& other) {
    if (other.state_ == State::MATERIALIZED) {
        if (other.is_constant_) {
            constant_data_ = other.constant_data_;
        } else {
            data_ = std::make_unique<float[]>(
                numel_);  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Dynamic array for tensor data
            if (other.data_) {
                std::copy(other.data_.get(), other.data_.get() + numel_, data_.get());
            }
        }
    } else {
        data_ = nullptr;
        constant_data_ = nullptr;
    }
}

void Tensor::move_from_other(
    Tensor&&
        other) {  // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - Function resets moved-from object to valid state
    // Move all members from other
    state_ = std::move(other.state_);
    producer_node_ = std::move(other.producer_node_);
    output_index_ = std::move(other.output_index_);
    rank_ = std::move(other.rank_);
    numel_ = std::move(other.numel_);
    is_constant_ = std::move(other.is_constant_);
    evaluation_in_progress_ = other.evaluation_in_progress_.load();

    // Move array data
    std::copy(other.shape_, other.shape_ + 4, shape_);

    if (other.state_ == State::MATERIALIZED) {
        if (other.is_constant_) {
            constant_data_ = other.constant_data_;
            data_ = nullptr;
        } else {
            data_ = std::move(other.data_);
            constant_data_ = nullptr;
        }
    } else {
        data_ = nullptr;
        constant_data_ = nullptr;
    }

    // Reset other tensor to valid state
    other.state_ = State::LAZY;
    other.producer_node_ = 0;
    other.output_index_ = 0;
    other.rank_ = 0;
    other.numel_ = 0;
    other.is_constant_ = false;
    other.constant_data_ = nullptr;
    other.evaluation_in_progress_ = false;
    std::fill(other.shape_, other.shape_ + 4, 1);
}

// Stream operator implementation
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    tensor.print_graph(os, 0);
    return os;
}
