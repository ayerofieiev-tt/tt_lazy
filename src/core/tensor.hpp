#pragma once
#include "op_args.hpp"
#include "shape.hpp"
#include "common.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

enum class DataType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT32,
    UINT64,
    BOOL
};

enum class TensorState {
    LAZY,      // Contains graph node information
    SCHEDULED, // Operation scheduled for evaluation
    EVALUATED  // Contains actual data
};

DEFINE_OP_ARGS(Input,
    // No additional arguments needed
);

class Tensor {
public:
    Tensor();
    Tensor(const Shape& shape, DataType dtype, std::shared_ptr<OpArgs> args, std::vector<Tensor> inputs, uint16_t output_index = 0);
    Tensor(const Shape& shape, std::vector<float>&& data);

    static std::vector<Tensor> make_outputs(
        const std::vector<Shape>& shapes,
        const std::vector<DataType>& dtypes,
        const std::shared_ptr<OpArgs>& args,
        const std::vector<Tensor>& inputs);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    TensorState state() const;
    bool is_lazy() const;
    bool is_evaluated() const;
    bool is_scheduled() const;

    uint16_t output_index() const;
    
    uintptr_t runtime_id() const;
    OpTypeId op_type_id() const;
    const OpArgs& op_args() const;
    OpArgs& op_args();
    std::string_view op_name() const;
    
    const std::vector<Tensor>& inputs() const;
    const std::vector<Tensor>& outputs() const;
    void set_outputs(std::vector<Tensor> outputs);

    const Shape& shape() const;
    uint16_t rank() const;
    uint32_t size(size_t dim) const;
    size_t total_elements() const;

    std::vector<float>& data();
    const std::vector<float>& data() const;
    
    size_t element_size() const;
    DataType dtype() const;

    // Data management methods
    void set_host_data(std::vector<float>&& data);
    bool has_host_data() const;

    void eval();

private:
    struct TensorSpec {
        TensorSpec() = default;
        
        TensorSpec(const Shape& shape, DataType dtype, 
                std::shared_ptr<OpArgs> op_args, uint16_t output_index = 0)
            : shape(shape), dtype(dtype), op_args(op_args),
            inputs(), outputs(), output_index(output_index), state(TensorState::LAZY) {
        }
            
        ~TensorSpec() {
            // Vector automatically cleans up its memory
        }
            
        // This data determines how to produce the tensor
        Shape shape;
        DataType dtype = DataType::FLOAT32;
        std::shared_ptr<OpArgs> op_args;
        std::vector<Tensor> inputs;
        
        // Results of evaluation
        std::vector<Tensor> outputs;
        uint16_t output_index = 0;
        
        // Whether the tensor is spec-ed, scheduled for evaluation or evaluated
        TensorState state = TensorState::LAZY;
        
        // Data storage - can be input data, output data, or device handle
        std::vector<float> host_data;           // Host memory for input/output data
    };

    std::shared_ptr<TensorSpec> spec_;
};