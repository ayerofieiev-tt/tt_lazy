#include "tensor.hpp"

#include "evaluation_manager.hpp"
#include "shape.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

#include <spdlog/spdlog.h>

Tensor::Tensor()
    : spec_(std::make_shared<TensorSpec>()) {
}

Tensor::Tensor(const Shape& shape, DataType dtype, std::shared_ptr<OpArgs> args, std::vector<Tensor> inputs, uint16_t output_index)
    : spec_(std::make_shared<TensorSpec>(shape, dtype, args, output_index)) {
    spec_->inputs = std::move(inputs);
}

Tensor::Tensor(const Shape& shape, std::vector<float>&& data)
    : spec_(std::make_shared<TensorSpec>(shape, DataType::FLOAT32, std::make_shared<OpArgs>(InputArgs{}), 0)) {
    spec_->host_data = std::move(data);
}

std::vector<Tensor> Tensor::make_outputs(
    const std::vector<Shape>& shapes,
    const std::vector<DataType>& dtypes,
    const std::shared_ptr<OpArgs>& args,
    const std::vector<Tensor>& inputs) {
    
    assert(shapes.size() == dtypes.size());
    
    std::vector<Tensor> outputs;
    outputs.reserve(shapes.size());
    
    for (size_t i = 0; i < shapes.size(); ++i) {        
        outputs.emplace_back(Tensor(shapes[i], dtypes[i], args, inputs, static_cast<uint16_t>(i)));
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto siblings = outputs;
        siblings.erase(siblings.begin() + static_cast<long>(i));
        outputs[i].set_outputs(std::move(siblings));     
    }
    
    return outputs;
}

Tensor::Tensor(const Tensor& other)
    : spec_(other.spec_) {
}

Tensor::Tensor(Tensor&& other) noexcept
    : spec_(std::move(other.spec_)) {
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        spec_ = other.spec_;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        spec_ = std::move(other.spec_);
    }
    return *this;
}

Tensor::~Tensor() = default;

void Tensor::eval() {
    if (is_evaluated()) {
        return;
    }

    if (is_scheduled()) {
        return;
    }

    spec_->state = TensorState::SCHEDULED;

    try {
        auto& eval_manager = tt_lazy::get_evaluation_manager();
        eval_manager.evaluate(*this);
    } catch (...) {
        // Handle evaluation failure - reset to lazy state
        spec_->state = TensorState::LAZY;
        throw;
    }
}

size_t Tensor::element_size() const {
    switch (spec_->dtype) {
        case DataType::FLOAT32:
            return sizeof(float);
        case DataType::FLOAT64:
            return sizeof(double);
        case DataType::INT32:
            return sizeof(int32_t);
        case DataType::INT64:
            return sizeof(int64_t);
        case DataType::UINT32:
            return sizeof(uint32_t);
        case DataType::UINT64:
            return sizeof(uint64_t);
        case DataType::BOOL:
            return sizeof(bool);
        default:
            throw std::runtime_error("Unknown data type");
    }
}

DataType Tensor::dtype() const {
    return spec_->dtype;
}

TensorState Tensor::state() const {
    return spec_->state;
}

bool Tensor::is_lazy() const {
    return spec_->state == TensorState::LAZY;
}

bool Tensor::is_evaluated() const {
    return spec_->state == TensorState::EVALUATED;
}

bool Tensor::is_scheduled() const {
    return spec_->state == TensorState::SCHEDULED;
}

uint16_t Tensor::output_index() const {
    return spec_->output_index;
}

uintptr_t Tensor::runtime_id() const {
    return reinterpret_cast<uintptr_t>(spec_.get());
}

OpTypeId Tensor::op_type_id() const {
    return spec_->op_args->type_id();
}

const OpArgs& Tensor::op_args() const {
    return *spec_->op_args;
}

OpArgs& Tensor::op_args() {
    return *spec_->op_args;
}

std::string_view Tensor::op_name() const {
    return spec_->op_args->op_name();
}

const std::vector<Tensor>& Tensor::inputs() const {
    return spec_->inputs;
}

const std::vector<Tensor>& Tensor::outputs() const {
    return spec_->outputs;
}

void Tensor::set_outputs(std::vector<Tensor> outputs) {    
    spec_->outputs = std::move(outputs);    
}

const Shape& Tensor::shape() const {
    return spec_->shape;
}

uint16_t Tensor::rank() const {
    return static_cast<uint16_t>(spec_->shape.rank());
}

uint32_t Tensor::size(size_t dim) const {
    return dim < spec_->shape.rank() ? spec_->shape[dim] : 1;
}

size_t Tensor::total_elements() const {
    return spec_->shape.total_elements();
}

std::vector<float>& Tensor::data() {
    return spec_->host_data;
}

const std::vector<float>& Tensor::data() const {
    return spec_->host_data;
}

void Tensor::set_host_data(std::vector<float>&& data) {
    spec_->host_data = std::move(data);
}

bool Tensor::has_host_data() const {
    return !spec_->host_data.empty();
}