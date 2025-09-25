# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Quick Development Build
```bash
./build.sh dev     # Development build with sanitizers, clang-tidy, and debug symbols
./build.sh release # Optimized release build
```

### Manual CMake (Advanced)
```bash
# Install dependencies
conan install . --build=missing -s build_type=Debug

# Configure and build using presets
cmake --preset dev && cmake --build --preset dev

# Run tests
ctest --preset dev --output-on-failure

# Available presets: dev, release
cmake --list-presets
```

### Code Quality Tools
```bash
# Run static analysis
./run-clang-tidy.sh

# Apply automatic fixes and parallel execution
./run-clang-tidy.sh --fix --jobs 8

# Pre-commit hooks (if installed)
pre-commit run --all-files
```

### Testing
```bash
# C++ tests
ctest --preset dev --output-on-failure

# Run specific test
./build-dev/tt_lazy_tests --gtest_filter="*TestName*"

# Run graph visualization demo
./build.sh dev  # Automatically builds and runs the demo

# Manual demo execution
./build-dev/graph_visualization_demo

# Python tests
cd tests/python && python3 run_tests.py
```

## Architecture Overview

TT Lazy is a **lazy evaluation C++ ML framework** that has been simplified to a clean 2-layer architecture:

### 1. Frontend Layer (`src/frontend/`)
- **Purpose**: User-facing operations that build computation graphs
- **Key**: Fast dispatch - operations build graph nodes without computation
- **Files**: `operations.hpp`, `operations.cpp`
- **Example**: `matmul(a, b)`, `relu(x)`, `reduce_sum(y)` build nodes instantly

### 2. Core Layer (`src/core/`)
- **Purpose**: Core infrastructure for tensor management and evaluation
- **Key Components**:
  - `tensor.hpp`: Unified lazy/materialized tensor class with graph visualization
  - `shape.hpp`: Tensor shape utilities and operations
  - `graph_utils.hpp`: Graph manipulation and traversal utilities
  - `evaluation_manager.hpp`: Handles tensor materialization and computation
  - `op_args.hpp`: Type-safe operation argument system
- **Key**: Lazy tensors store graph references until materialization triggers evaluation

### Examples and Demos
- `examples/graph_visualization_demo.cpp`: Demonstrates tensor graph construction and visualization
- **Auto-built**: The demo is automatically built and run during development builds

## Key Development Patterns

### Adding New Operations (Simplified Pattern)

1. **Define Operation Arguments** (`src/core/op_args.hpp`):
```cpp
DEFINE_OP_ARGS(NewOp,
    bool param = false;
);
```

2. **Implement Frontend Operation** (`src/frontend/operations.hpp`, `src/frontend/operations.cpp`):
```cpp
// Declare in operations.hpp
Tensor new_op(const Tensor& input, bool param = false);

// Implement in operations.cpp
Tensor new_op(const Tensor& input, bool param) {
    NewOpArgs args{param};
    // Create operation node in computation graph
    // Implementation details handled by evaluation manager
    return create_operation_tensor<NewOpArgs>(args, {input});
}
```

3. **Handle Evaluation** (`src/core/evaluation_manager.cpp`):
```cpp
// Add evaluation logic for new operation type
// The evaluation manager handles materialization and computation
```

### Tensor Lifecycle (Simplified)
1. **Creation**: `Tensor c = matmul(a, b)` creates lazy tensor with graph reference
2. **Graph Building**: Operations chain without computation, building graph structure
3. **Materialization**: Triggered by data access (`data_ptr()`, `to_vector()`, `to_string()`)
4. **Evaluation**: EvaluationManager handles computation and caching
5. **Result**: Materialized tensor with actual data and preserved graph info

### Memory Management
- **Lazy tensors**: Store graph references and shape metadata
- **Materialized tensors**: Contain actual computed data
- **Graph visualization**: Available for both lazy and materialized tensors
- **Evaluation**: Automatic on data access, explicit with `eval()`

## Code Conventions

### Naming
- Classes: `CamelCase` (Tensor, TapeExecutor)
- Functions: `snake_case` (matmul, reduce_sum)
- Variables: `snake_case` with trailing `_` for private members
- Constants: `UPPER_CASE`

### Libraries Structure
- `tt_lazy_core`: Basic infrastructure (tensor, shape, graph utilities, evaluation)
- `tt_lazy_operations`: Frontend operations building graphs
- `tt_lazy_lib`: Combined interface library (core + operations)
- Note: Tape and math layers removed in recent simplification

### Build Configuration
- **Development builds**: Enable ASAN+UBSAN sanitizers and clang-tidy by default
- **Release builds**: Optimized with all safety checks disabled
- **Ninja**: Default generator for fast parallel builds
- **Conan**: Handles all dependencies (Boost, GTest, pybind11, spdlog)
- **Logging**: Uses spdlog for structured logging and debugging

### Testing Strategy
- **Unit tests**: Component-focused tests in `tests/cpp/unit/`
  - `unit/test_tensor.cpp`: Core tensor functionality and graph visualization
- **Integration tests**: Cross-component tests in `tests/cpp/integration/`
  - `integration/test_operations.cpp`: Frontend operation chains
  - `integration/test_end_to_end.cpp`: Full pipeline testing
- **Benchmarks**: Performance tests in `tests/cpp/benchmarks/`
  - `benchmarks/test_mlp_demo.cpp`: Complex graph scenarios
- **Examples**: Interactive demos in `examples/`
  - `graph_visualization_demo.cpp`: Live graph construction demonstration

## Important Implementation Notes

### Operation Arguments
Use `DEFINE_OP_ARGS` macro for type-safe operation parameters:
```cpp
DEFINE_OP_ARGS(MatMul,
    bool transpose_a = false;
    bool transpose_b = false;
);
```

### Graph Visualization and Debugging
- **Graph visualization**: Use `tensor.to_string()` or `std::cout << tensor` for graph inspection
- **Spdlog integration**: Use `spdlog::info("{}", tensor.to_string())` for structured logging
- **Demo integration**: `graph_visualization_demo` shows live graph construction

### Error Handling
- Use exceptions for error conditions with descriptive messages
- Validate tensor shapes and arguments in evaluation manager
- Check materialization status when accessing data

### Performance Considerations
- Frontend operations must be fast (graph building only)
- EvaluationManager handles computation optimization and caching
- Graph visualization is available without performance penalty
- Materialization triggers computation - use strategically
