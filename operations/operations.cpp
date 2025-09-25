#include "operations.hpp"

#include <algorithm>

// Helper to create tensors from node with multiple outputs
std::vector<Tensor> make_output_tensors(
    NodeId producer_node_id, size_t output_count,
    const std::vector<std::vector<uint32_t>>&
        shapes) {  // NOLINT(bugprone-easily-swappable-parameters) - Semantically different parameters
    std::vector<Tensor> outputs;
    outputs.reserve(output_count);

    for (size_t i = 0; i < output_count; ++i) {
        const auto& shape =
            i < shapes.size()
                ? shapes[i]
                : std::vector<uint32_t>{
                      1};  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
        // Convert vector to initializer_list format
        assert(shape.size() <= 4);
        uint32_t shape_array[4] = {
            1, 1, 1,
            1};  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Temporary array for shape conversion // NOLINT(cppcoreguidelines-avoid-c-arrays) - Temporary array for shape conversion
        for (size_t j = 0; j < shape.size(); ++j) {
            shape_array[j] = shape
                [j];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
        }
        Tensor tensor(producer_node_id, static_cast<uint16_t>(i),
                      {shape_array[0], shape_array[1], shape_array[2], shape_array[3]});
        outputs.push_back(tensor);
    }

    return outputs;
}

// Split function - implicit graph building!
std::vector<Tensor> split(const Tensor& input, int64_t split_size, int32_t dim) {
    // Create operation arguments
    SplitArgs args;
    args.split_size = split_size;
    args.dim = dim;

    // Create input list
    SmallVector<Tensor, 2> inputs{input};

    // Create node in global context
    NodeId node_id = Context::instance().create_node(inputs, std::move(args));

    // Calculate output shapes (simplified - you'd compute real shapes based on input)
    size_t input_size = static_cast<size_t>(input.size(static_cast<size_t>(dim)));
    size_t num_outputs = (input_size + static_cast<size_t>(split_size) - 1) / static_cast<size_t>(split_size);

    std::vector<std::vector<uint32_t>> output_shapes;
    for (size_t i = 0; i < num_outputs; ++i) {
        // Copy input shape but adjust split dimension
        std::vector<uint32_t> shape(input.shape(), input.shape() + input.rank());
        if (dim >= 0 && static_cast<size_t>(dim) < shape.size()) {
            shape[static_cast<size_t>(dim)] = static_cast<uint32_t>(
                std::min(split_size,
                         static_cast<int64_t>(input_size - static_cast<size_t>(i) * static_cast<size_t>(split_size))));
        }
        output_shapes.push_back(shape);
    }

    return make_output_tensors(node_id, num_outputs, output_shapes);
}

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b) {
    MatMulArgs args;
    args.transpose_a = transpose_a;
    args.transpose_b = transpose_b;

    SmallVector<Tensor, 2> inputs{a, b};

    NodeId node_id = Context::instance().create_node(inputs, std::move(args));

    // Calculate output shape (simplified)
    uint32_t rows = transpose_a ? a.size(1) : a.size(0);
    uint32_t cols = transpose_b ? b.size(0) : b.size(1);

    return Tensor(node_id, 0, {rows, cols});
}

Tensor reduce_sum(const Tensor& input, const std::vector<int32_t>& dims, bool keepdim) {
    ReduceArgs args;
    for (int32_t dim : dims) {
        args.dims.push_back(dim);
    }
    args.keepdim = keepdim;
    args.type = ReduceArgs::Type::SUM;

    SmallVector<Tensor, 2> inputs{input};

    NodeId node_id = Context::instance().create_node(inputs, std::move(args));

    // Calculate output shape (simplified)
    std::vector<uint32_t> output_shape;
    for (size_t i = 0; i < input.rank(); ++i) {
        bool is_reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
        if (!is_reduced || keepdim) {
            output_shape.push_back(is_reduced ? 1 : input.size(i));
        }
    }

    // Convert vector to initializer_list format
    assert(output_shape.size() <= 4);
    uint32_t shape_array[4] = {1, 1, 1,
                               1};  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Temporary array for shape conversion
    for (size_t i = 0; i < output_shape.size(); ++i) {
        shape_array[i] = output_shape
            [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
    }
    return Tensor(node_id, 0, {shape_array[0], shape_array[1], shape_array[2], shape_array[3]});
}

Tensor relu(const Tensor& input) {
    ReLUArgs args;
    args.inplace = false;

    SmallVector<Tensor, 2> inputs{input};

    NodeId node_id = Context::instance().create_node(inputs, std::move(args));

    // Output has same shape as input
    std::vector<uint32_t> shape(input.shape(), input.shape() + input.rank());

    // Convert vector to initializer_list format
    assert(shape.size() <= 4);
    uint32_t shape_array[4] = {1, 1, 1,
                               1};  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Temporary array for shape conversion
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_array[i] = shape
            [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
    }
    return Tensor(node_id, 0, {shape_array[0], shape_array[1], shape_array[2], shape_array[3]});
}

Tensor add(const Tensor& a, const Tensor& b) {
    AddArgs args;

    SmallVector<Tensor, 2> inputs{a, b};

    NodeId node_id = Context::instance().create_node(inputs, std::move(args));

    // Output shape is broadcasted shape of inputs
    std::vector<uint32_t> a_shape(a.shape(), a.shape() + a.rank());
    std::vector<uint32_t> b_shape(b.shape(), b.shape() + b.rank());
    auto output_shape = Tensor::broadcast_shapes(a_shape, b_shape);

    // Convert vector to initializer_list format
    assert(output_shape.size() <= 4);
    uint32_t shape_array[4] = {1, 1, 1,
                               1};  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Temporary array for shape conversion
    for (size_t i = 0; i < output_shape.size(); ++i) {
        shape_array[i] = output_shape
            [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
    }
    return Tensor(node_id, 0, {shape_array[0], shape_array[1], shape_array[2], shape_array[3]});
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    MultiplyArgs args;

    SmallVector<Tensor, 2> inputs{a, b};

    NodeId node_id = Context::instance().create_node(inputs, std::move(args));

    // Output shape is broadcasted shape of inputs
    std::vector<uint32_t> a_shape(a.shape(), a.shape() + a.rank());
    std::vector<uint32_t> b_shape(b.shape(), b.shape() + b.rank());
    auto output_shape = Tensor::broadcast_shapes(a_shape, b_shape);

    // Convert vector to initializer_list format
    assert(output_shape.size() <= 4);
    uint32_t shape_array[4] = {1, 1, 1,
                               1};  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Temporary array for shape conversion
    for (size_t i = 0; i < output_shape.size(); ++i) {
        shape_array[i] = output_shape
            [i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) - Safe array access with bounds checking
    }
    return Tensor(node_id, 0, {shape_array[0], shape_array[1], shape_array[2], shape_array[3]});
}

Tensor fused_mlp(const Tensor& input, const Tensor& weights, const Tensor& bias, bool has_relu) {
    FusedMLPArgs args;
    args.has_relu = has_relu;
    args.fusion_info = std::string("MatMul + Add") + (has_relu ? " + ReLU" : "");

    // Use 3 inputs: input, weights, and bias - much cleaner!
    SmallVector<Tensor, 3> inputs{input, weights, bias};

    NodeId node_id = Context::instance().create_node(inputs, std::move(args));

    // Store bias as additional data - the tape generator will handle this
    // For now, we assume the operation handler will get bias from the context

    // Output shape calculation: input @ weights + bias
    // input: [batch_size, input_features]
    // weights: [input_features, output_features]
    // bias: [1, output_features]
    // output: [batch_size, output_features]

    uint32_t batch_size = input.size(0);
    uint32_t output_features = weights.size(1);

    return Tensor(node_id, 0, {batch_size, output_features});
}
