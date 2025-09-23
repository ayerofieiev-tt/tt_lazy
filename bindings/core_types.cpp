#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Tensor.hpp"
#include "Node.hpp"
#include "Context.hpp"

namespace py = pybind11;

void bind_core_types(py::module& m) {
    // Tensor class
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>(), "Create a null tensor")
        .def(py::init<NodeId, uint16_t, std::initializer_list<uint32_t>>(), 
             py::arg("producer"), py::arg("output_idx"), py::arg("shape"), "Create a tensor from a node")
        .def("rank", &Tensor::rank, "Get tensor rank")
        .def("size", &Tensor::size, py::arg("dim"), "Get size of dimension")
        .def("is_constant", &Tensor::is_constant, "Check if tensor is constant")
        .def("producer_node", &Tensor::producer_node, "Get producer node ID")
        .def("output_index", &Tensor::output_index, "Get output index")
        .def("shape", [](const Tensor& t) {
            std::vector<uint32_t> shape;
            for (uint16_t i = 0; i < t.rank(); ++i) {
                shape.push_back(t.size(i));
            }
            return shape;
        }, "Get tensor shape as list");

    // Node class
    py::class_<Node>(m, "Node")
        .def("id", &Node::id, "Get node ID")
        .def("op_name", &Node::op_name, "Get operation name")
        .def("type_id", &Node::type_id, "Get operation type ID")
        .def("inputs", &Node::inputs, "Get input tensors")
        .def("output_nodes", &Node::output_nodes, "Get output node IDs");

    // Context class
    py::class_<Context>(m, "Context")
        .def_static("instance", &Context::instance, 
                   py::return_value_policy::reference, "Get global context instance")
        .def("size", &Context::size, "Get number of nodes")
        .def("clear", &Context::clear, "Clear all nodes")
        .def("print_stats", &Context::print_stats, "Print context statistics");

    // Utility functions
    m.def("create_constant_tensor", [](py::array_t<float> data, 
                                      const std::vector<uint32_t>& shape) {
        if (data.ndim() != static_cast<int>(shape.size())) {
            throw std::runtime_error("Data dimensions don't match shape");
        }
        // Convert vector to initializer_list manually
        uint32_t shape_array[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < shape.size() && i < 4; ++i) {
            shape_array[i] = shape[i];
        }
        return Tensor(static_cast<void*>(data.mutable_data()), 
                            {shape_array[0], shape_array[1], shape_array[2], shape_array[3]});
    }, py::arg("data"), py::arg("shape"), "Create a constant tensor from numpy array");
}
