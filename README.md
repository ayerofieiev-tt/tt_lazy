# TT Lazy

A high-performance C++ machine learning framework with lazy evaluation, similar to MLX, designed for fast dispatch times and efficient computation graphs.

## 🏗️ Core Architecture

TT Lazy is a **CPU math functions backend** with a **lazy tensor evaluation framework**. The system stores operation graphs and optimizes them before computation, similar to MLX but with a focus on fast dispatch times.


### Core Concept

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
│  User Operations: matmul(a,b), relu(x), reduce_sum(y)      │
│  • Fast dispatch (just graph building)                     │
│  • No computation, only graph construction                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         GRAPH                               │
│  Computation Graph: Nodes + Dependencies                    │
│  • Lazy tensors store graph references                     │
│  • Operations stored as nodes with arguments               │
│  • No actual computation yet                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                          TAPE                               │
│  Linear Execution Plan + Optimization                       │
│  • Dead code elimination                                   │
│  • Operation fusion (future)                               │
│  • Memory optimization (future)                            │
│  • Operation handlers (bridge to math)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                          MATH                               │
│  CPU Math Functions: Actual Computation                     │
│  • Element-wise operations (ReLU, Sigmoid)                 │
│  • Matrix operations (MatMul, Transpose)                   │
│  • Reduction operations (Sum, Mean)                        │
│  • Memory-efficient implementations                        │
└─────────────────────────────────────────────────────────────┘
```

### Libraries

- **tt_lazy_core**: Basic graph infrastructure (Tensor, Node, Context, MemoryManager)
- **tt_lazy_operations**: Frontend operations that build computation graphs (Split, MatMul, Reduce, ReLU)
- **tt_math_lib**: CPU math functions for actual computation (immediate evaluation)
- **tt_lazy_tape**: Tape-based execution system with operation handlers (lowering/bridge layer)

## 🚀 Quick Start & Usage

### C++ API - Automatic Evaluation

```cpp
#include "tensor.hpp"
#include "operations.hpp"

// Create tensors
Tensor a({2, 3});
Tensor b({3, 4});
a.fill(1.0f);
b.fill(2.0f);

// Build lazy computation graph (no computation yet!)
Tensor c = matmul(a, b);      // Lazy operation
Tensor d = relu(c);           // Lazy operation  
Tensor e = reduce_sum(d);     // Lazy operation

// Automatic evaluation when accessing data
float* result = e.data_ptr(); // Graph evaluated automatically!
std::vector<float> data = e.to_vector(); // Also triggers evaluation
```

### Advanced Usage with Graph Optimization

```cpp
// Multiple element-wise operations that get fused
Tensor x({1000, 1000});
x.fill(1.0f);

Tensor y = relu(x);                   // Element-wise
Tensor z = add(y, y);                 // Element-wise  
Tensor w = multiply(z, z);            // Element-wise

// TT Lazy optimizes: relu + add + multiply → single fused kernel
float* optimized_result = w.data_ptr(); // Fused execution!
```

### Python API

```python
import tt_lazy
import numpy as np

# Create tensors
a = tt_lazy.tensor([2, 3], data=np.ones((2, 3), dtype=np.float32))
b = tt_lazy.tensor([3, 4], data=np.ones((3, 4), dtype=np.float32))

# Build lazy computation graph
c = tt_lazy.matmul(a, b)        # No computation yet
d = tt_lazy.relu(c)             # Still no computation
e = tt_lazy.reduce_sum(d)       # Still lazy

# Automatic evaluation when converting to numpy
result_np = e.to_numpy()        # Graph evaluated automatically!
```

### Debugging

```cpp
// Print tensor information
Tensor result = some_computation();
std::cout << result << std::endl; // Stream output
spdlog::info("{}", result.to_string()); // Or use to_string() directly

// Manual evaluation when needed
result.eval(); // Explicit evaluation (optional)
```

## 📦 Dependencies

- **C++17** or later
- **CMake** 3.16+
- **Conan** 2.0+ (for dependency management)
- **Boost** 1.84.0+ (container library)
- **Google Test** 1.14.0+ (for testing)
- **pybind11** 2.12.0+ (for Python bindings)

## 🛠️ Installation

### Prerequisites

1. Install Conan:
```bash
pip install conan
```

2. Ensure Conan is in your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"  # For pipx installations
```

### Build

TT Lazy uses **Ninja** as the default build system for fast parallel builds and CMake presets for streamlined configuration.

#### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd tt_lazy

# Build with Conan (recommended)
chmod +x build_with_conan.sh
./build_with_conan.sh
```

#### Build Options

**Release Build (Production):**
```bash
./build_with_conan.sh           # Full build with dependencies
```

**Debug Build:**
```bash
./build_debug.sh                # Debug build with all debug symbols
```

**CMake Presets (Advanced):**
```bash
# Using CMake presets directly
cmake --preset conan-release    # Configure
cmake --build --preset conan-release  # Build
ctest --preset conan-release    # Test

# Available presets:
cmake --list-presets            # See all available presets
```

#### What the build does:
- Install all dependencies via Conan
- Configure CMake with Ninja generator
- Build all libraries and tests with parallel compilation
- Run the complete test suite
- Generate compile commands for IDE support

### Manual Build

```bash
# Install dependencies
conan install . --build=missing

# Configure with preset
cmake --preset conan-release

# Build with Ninja
cmake --build --preset conan-release

# Run tests
ctest --preset conan-release
```

#### Prerequisites for Ninja

**macOS:**
```bash
brew install ninja
```

**Ubuntu/Debian:**
```bash
sudo apt install ninja-build
```

**Windows:**
```bash
choco install ninja          # Using Chocolatey
# OR download from: https://ninja-build.org/
```

## 🧪 Testing

### C++ Tests
```bash
cd build
ctest --output-on-failure
```

### Python Tests
```bash
cd tests/python
python3 run_tests.py
```


## 🔧 Operations

### Core Operations

- **MatMul**: Matrix multiplication with optional transposition
- **ReLU**: Rectified Linear Unit activation
- **Reduce**: Sum, mean, max, min along specified dimensions
- **Split**: Split tensor along a dimension
- **Add/Multiply**: Element-wise operations
- **Transpose**: Transpose tensor dimensions

### Operation Arguments

Operations support configurable arguments:

```cpp
// Matrix multiplication with transposition
Tensor result = matmul(a, b, true, false);  // transpose_a=true, transpose_b=false

// Reduce with specific dimensions
Tensor sum = reduce_sum(input, {0, 2}, true);  // dims={0,2}, keepdim=true

// ReLU in-place
Tensor activated = relu(input, true);  // inplace=true
```

## 🛠️ Adding New Operations

Adding a new operation requires implementing three layers: **Frontend**, **Math**, and **Handler**.

### 1. Frontend Operation (Graph Building)

**File**: `includes/operations/operations.hpp` and `frontend/operations.cpp`

```cpp
// 1. Define operation arguments
DEFINE_OP_ARGS(Sigmoid,
    bool inplace = false;
);

// 2. Declare frontend function
Tensor sigmoid(const Tensor& input, bool inplace = false);

// 3. Implement frontend function (builds graph)
Tensor sigmoid(const Tensor& input, bool inplace) {
    SigmoidArgs args;
    args.inplace = inplace;
    
    SmallVector<Tensor, 2> inputs{input};
    NodeId node_id = Context::instance().create_node(inputs, std::move(args));
    
    // Output has same shape as input
    std::vector<uint32_t> shape(input.shape(), input.shape() + input.rank());
    uint32_t shape_array[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_array[i] = shape[i];
    }
    return Tensor(node_id, 0, {shape_array[0], shape_array[1], shape_array[2], shape_array[3]});
}
```

### 2. Math Function (CPU Implementation)

**File**: `math/math_operations.hpp` and `math/eltwise.cpp` (or new file)

```cpp
// 1. Declare in math_operations.hpp
namespace math {
    Tensor sigmoid(const Tensor& input);
}

// 2. Implement in math/eltwise.cpp
namespace math {
    Tensor sigmoid(const Tensor& input) {
        // Ensure input is materialized
        if (!input.is_materialized()) {
            throw std::runtime_error("Math functions require materialized tensors");
        }
        
        // Create output tensor with same shape
        Tensor output(input.shape(), input.shape() + input.rank());
        
        // Perform actual computation
        const float* input_data = input.const_data_ptr();
        float* output_data = output.data_ptr();
        size_t num_elements = input.total_elements();
        
        for (size_t i = 0; i < num_elements; ++i) {
            output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
        }
        
        return output;
    }
}
```

### 3. Operation Handler (Bridge/Lowering)

**File**: `tape/OperationHandlers.cpp`

```cpp
// 1. Implement handler function
void handle_sigmoid(TapeOperation& op, TapeExecutor& executor) {
    // Collect input tensors
    std::vector<std::shared_ptr<Tensor>> input_tensors;
    
    // Add lazy input tensors
    for (NodeId node_id : op.input_nodes) {
        auto tensor = executor.get_result(node_id);
        if (!tensor) {
            throw std::runtime_error("Missing lazy input tensor for sigmoid operation");
        }
        input_tensors.push_back(tensor);
    }
    
    // Add constant input tensors
    for (const auto& const_tensor : op.constant_inputs) {
        input_tensors.push_back(std::make_shared<Tensor>(const_tensor));
    }
    
    if (input_tensors.size() != 1) {
        throw std::runtime_error("Sigmoid operation requires exactly 1 input");
    }
    
    // Call math function
    auto result = std::make_shared<Tensor>(math::sigmoid(*input_tensors[0]));
    executor.set_result(op.node_id, result);
    op.result = result;
}

// 2. Register handler in register_all_operations()
void register_all_operations(TapeExecutor& executor) {
    executor.register_operation(SplitArgs::type_id(), handle_split);
    executor.register_operation(MatMulArgs::type_id(), handle_matmul);
    executor.register_operation(ReduceArgs::type_id(), handle_reduce);
    executor.register_operation(ReLUArgs::type_id(), handle_relu);
    executor.register_operation(SigmoidArgs::type_id(), handle_sigmoid);  // Add this line
}
```

### 4. Python Bindings (Optional)

**File**: `bindings/operations.cpp`

```cpp
// Add Python binding
m.def("sigmoid", &sigmoid, "Apply sigmoid activation", 
      py::arg("input"), py::arg("inplace") = false);
```

### Complete Example: Sigmoid Operation

```cpp
// Usage in C++
Tensor x({2, 3});
x.fill(0.5f);
Tensor y = sigmoid(x);        // Lazy operation - builds graph
float* data = y.data_ptr();   // Automatic evaluation!

// Usage in Python
import tt_lazy
x = tt_lazy.tensor([2, 3], data=[[0.5, 1.0, -1.0], [2.0, -0.5, 0.0]])
y = tt_lazy.sigmoid(x)        # Lazy operation
result = y.to_numpy()         # Automatic evaluation!
```

### Operation Flow Summary

1. **Frontend**: `sigmoid(x)` creates graph node with `SigmoidArgs`
2. **Lazy**: Operation stored in graph, no computation yet
3. **Materialization**: Tape executor processes graph
4. **Handler**: `handle_sigmoid()` bridges graph operation to math function
5. **Math**: `math::sigmoid()` performs actual CPU computation
6. **Result**: Materialized tensor returned to user

## 🏗️ Project Structure

```
tt_lazy/
├── core/                   # Core source files
│   ├── Tensor.cpp         # Tensor implementation
│   ├── Node.cpp           # Graph node implementation
│   ├── Context.cpp        # Global context
│   └── MemoryManager.cpp  # Memory management
├── includes/              # Header files
│   ├── Tensor.hpp         # Tensor interface
│   ├── Node.hpp           # Node interface
│   ├── Context.hpp        # Context interface
│   ├── operations/        # Operation definitions
│   └── tape/              # Tape system headers
├── frontend/              # Graph-based operations
├── math/                  # Immediate computation operations
├── tape/                  # Tape execution system
├── bindings/              # Python bindings
├── tests/                 # Test suite
│   ├── cpp/              # C++ unit tests
│   └── python/           # Python integration tests
├── build/                 # Build artifacts
├── CMakeLists.txt         # CMake configuration
├── conanfile.py          # Conan dependencies
└── build_with_conan.sh   # Build script
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
