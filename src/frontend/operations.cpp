#include "operations.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>

std::vector<Tensor> split(const Tensor& input, uint32_t split_size, uint32_t dim) {
    const uint32_t input_size = input.size(dim);
    const size_t num_outputs = (input_size + split_size - 1) / split_size;

    std::vector<Shape> output_shapes;
    for (size_t i = 0; i < num_outputs; ++i) {
        std::vector<uint32_t> shape = input.shape().to_vector();
        if (dim < shape.size()) {
            const uint32_t remaining = input_size - static_cast<uint32_t>(i) * split_size;
            shape[dim] = std::min(split_size, remaining);
        }
        output_shapes.push_back(Shape(shape));
    }
    
    auto shared_args = std::make_shared<OpArgs>(SplitArgs{.split_size = split_size, .dim = dim});
    std::vector<DataType> dtypes(num_outputs, input.dtype());
    std::vector<Tensor> inputs {input};
    
    return Tensor::make_outputs(output_shapes, dtypes, shared_args, inputs);
}

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b) {
    uint32_t rows = transpose_a ? a.size(1) : a.size(0);
    uint32_t cols = transpose_b ? b.size(0) : b.size(1);
    Shape output_shape({rows, cols});

    return Tensor(
        Shape(output_shape), 
        DataType::FLOAT32, 
        std::make_shared<OpArgs>(MatMulArgs{.transpose_a = transpose_a, .transpose_b = transpose_b}), 
        {a, b});
}

Tensor reduce_sum(const Tensor& input, const std::vector<int32_t>& dims, bool keepdim) {
    std::vector<uint32_t> output_shape;
    for (size_t i = 0; i < input.rank(); ++i) {
        bool is_reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
        if (!is_reduced || keepdim) {
            output_shape.push_back(is_reduced ? 1 : input.size(i));
        }
    }

    return Tensor(
        Shape(output_shape), 
        DataType::FLOAT32, 
        std::make_shared<OpArgs>(ReduceArgs{.dims = SmallVector<int32_t, 4>(dims.begin(), dims.end()), .keepdim = keepdim, .type = ReduceArgs::Type::SUM}), 
        {input});
}

Tensor relu(const Tensor& input) {
    return Tensor(
        input.shape(), 
        DataType::FLOAT32, 
        std::make_shared<OpArgs>(ReLUArgs{.inplace = false}), 
        {input});
}

Tensor add(const Tensor& a, const Tensor& b) {
    Shape output_shape = Shape::broadcast_shapes(a.shape(), b.shape());    
    return Tensor(output_shape, DataType::FLOAT32, std::make_shared<OpArgs>(AddArgs{}), {a, b});
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    auto output_shape = Shape::broadcast_shapes(a.shape(), b.shape());
    return Tensor(output_shape, DataType::FLOAT32, std::make_shared<OpArgs>(MultiplyArgs{}), {a, b});
}

Tensor fused_mlp(const Tensor& input, const Tensor& weights, const Tensor& bias, bool has_relu) {
    return Tensor(
        input.shape(), 
        DataType::FLOAT32, 
        std::make_shared<OpArgs>(FusedMLPArgs{.has_relu = has_relu, .fusion_info = std::string("MatMul + Add") + (has_relu ? " + ReLU" : "")}), 
        {input, weights, bias});
}

Tensor zeros(const Shape& shape) {
    size_t total_elements = shape.total_elements();
    std::vector<float> data(total_elements, 0.0f);
    return Tensor(shape, std::move(data));
}

Tensor ones(const Shape& shape) {
    size_t total_elements = shape.total_elements();
    std::vector<float> data(total_elements, 1.0f);
    return Tensor(shape, std::move(data));
}

Tensor rand(const Shape& shape) {
    size_t total_elements = shape.total_elements();
    std::vector<float> data(total_elements);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (size_t i = 0; i < total_elements; ++i) {
        data[i] = dis(gen);
    }
    
    return Tensor(shape, std::move(data));
}

