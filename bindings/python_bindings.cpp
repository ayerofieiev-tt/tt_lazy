#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void bind_core_types(py::module& m);
void bind_operations(py::module& m);

PYBIND11_MODULE(tt_lazy, m) {
    m.doc() = "TT Lazy - High-performance C++ ML framework with lazy evaluation";

    // Bind core types (Tensor, Node, Context)
    bind_core_types(m);
    
    // Bind operations (matmul, relu, split, reduce_sum)
    bind_operations(m);
}
