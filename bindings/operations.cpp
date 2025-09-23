#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "operations.hpp"

namespace py = pybind11;

void bind_operations(py::module& m) {
    // Graph operations
    m.def("matmul", &matmul, 
          py::arg("a"), py::arg("b"), 
          py::arg("transpose_a") = false, py::arg("transpose_b") = false, 
          "Matrix multiplication");

    m.def("relu", &relu, py::arg("input"), "ReLU activation");

    m.def("split", &split, 
          py::arg("input"), py::arg("split_size"), py::arg("dim") = 0, 
          "Split tensor");

    m.def("reduce_sum", &reduce_sum, 
          py::arg("input"), py::arg("dims") = std::vector<int32_t>{}, py::arg("keepdim") = false, 
          "Reduce tensor sum");
}
