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

# Python tests
cd tests/python && python3 run_tests.py
```

## Architecture Overview

TT Lazy is a **lazy evaluation C++ ML framework** with a 4-layer architecture. The project has been recently restructured with a clean `src/` directory organization that mirrors the architectural layers:

### 1. Frontend Layer (`src/frontend/`)
- **Purpose**: User-facing operations that build computation graphs
- **Key**: Fast dispatch - operations only create graph nodes, no computation
- **Example**: `matmul(a, b)`, `relu(x)`, `reduce_sum(y)` build nodes instantly

### 2. Core Layer (`src/core/`)
- **Purpose**: Core infrastructure storing operation graphs
- **Key Components**:
  - `Tensor`: Unified lazy/materialized tensor class
  - `Context`: Global graph state management
  - `MemoryManager`: Handles tensor data allocation
  - `Shape`: Tensor shape utilities
- **Key**: Lazy tensors store graph references until materialization

### 3. Tape Layer (`src/tape/`)
- **Purpose**: Converts graphs to linear execution plans with optimization
- **Key Components**:
  - `TapeExecutor`: Executes operation sequence
  - `OperationHandlers`: Bridge between graph ops and math functions
  - `passes/MLPFusionPass`: Fuses matrix operations for performance
- **Key**: Optimization passes run before execution

### 4. Backend Layer (`src/backend/cpu/`)
- **Purpose**: CPU implementations of actual computations
- **Key**: Functions require materialized tensors, perform immediate evaluation
- **Pattern**: Each operation has corresponding `math::function_name()`

## Key Development Patterns

### Adding New Operations (3-Step Pattern)

1. **Frontend** (`src/frontend/operations.hpp`, `src/frontend/operations.cpp`):
```cpp
// Define arguments
DEFINE_OP_ARGS(NewOp, bool param = false;);

// Implement graph builder
Tensor new_op(const Tensor& input, bool param = false) {
    NewOpArgs args{param};
    SmallVector<Tensor, 2> inputs{input};
    NodeId node_id = Context::instance().create_node(inputs, std::move(args));
    return Tensor(node_id, 0, input.shape());
}
```

2. **Backend** (`src/backend/cpu/math_operations.hpp`, `src/backend/cpu/eltwise.cpp`):
```cpp
namespace math {
    Tensor new_op(const Tensor& input) {
        // Requires materialized tensor, performs computation
        if (!input.is_materialized()) throw std::runtime_error("...");
        // ... actual computation
    }
}
```

3. **Handler** (`src/tape/OperationHandlers.cpp`):
```cpp
void handle_new_op(TapeOperation& op, TapeExecutor& executor) {
    // Bridge: collect inputs, call math function, store result
    auto result = std::make_shared<Tensor>(math::new_op(*input_tensors[0]));
    executor.set_result(op.node_id, result);
}

// Register in register_all_operations()
executor.register_operation(NewOpArgs::type_id(), handle_new_op);
```

### Tensor Lifecycle
1. **Creation**: `Tensor c = matmul(a, b)` creates lazy tensor (just graph node)
2. **Graph Building**: Operations chain without computation
3. **Materialization**: Triggered by `data_ptr()`, `to_vector()`, `to_numpy()`
4. **Execution**: Tape system optimizes and runs math functions
5. **Result**: Materialized tensor with actual data

### Memory Management
- **Lazy tensors**: Store `NodeId` + metadata, no data allocation
- **Materialized tensors**: Own data via `MemoryManager`
- **Evaluation**: Automatic on data access, manual with `eval()`

## Code Conventions

### Naming
- Classes: `CamelCase` (Tensor, TapeExecutor)
- Functions: `snake_case` (matmul, reduce_sum)
- Variables: `snake_case` with trailing `_` for private members
- Constants: `UPPER_CASE`

### Libraries Structure
- `tt_lazy_core`: Basic infrastructure (Tensor, Node, Context, MemoryManager)
- `tt_lazy_operations`: Frontend operations building graphs
- `tt_math_lib`: CPU math functions for computation
- `tt_lazy_tape`: Tape execution with optimization passes

### Build Configuration
- **Development builds**: Enable ASAN+UBSAN sanitizers and clang-tidy by default
- **Release builds**: Optimized with all safety checks disabled
- **Ninja**: Default generator for fast parallel builds
- **Conan**: Handles all dependencies (Boost, GTest, pybind11, spdlog)
- **Logging**: Uses spdlog for structured logging and debugging

### Testing Strategy
- **Unit tests**: Component-focused tests in `tests/cpp/unit/`
  - `unit/test_tensor.cpp`: Core tensor functionality
  - `unit/math/test_math_ops.cpp`: Backend math operations
- **Integration tests**: Cross-component tests in `tests/cpp/integration/`
  - `integration/test_operations.cpp`: Frontend operation chains
  - `integration/test_end_to_end.cpp`: Full pipeline testing
- **Benchmarks**: Performance tests in `tests/cpp/benchmarks/`
  - `benchmarks/test_mlp_demo.cpp`: MLP fusion and optimization
- **Python tests**: API compatibility tests in `tests/python/`

## Important Implementation Notes

### Operation Arguments
Use `DEFINE_OP_ARGS` macro for type-safe operation parameters:
```cpp
DEFINE_OP_ARGS(MatMul,
    bool transpose_a = false;
    bool transpose_b = false;
);
```

### Error Handling
- Math functions must check `input.is_materialized()`
- Use exceptions for error conditions with descriptive messages
- Validate tensor shapes and arguments in operation handlers

### Performance Considerations
- Frontend operations must be fast (graph building only)
- Math functions are the performance bottlenecks
- Tape optimization passes can fuse operations (see `MLPFusionPass`)
- Materialization triggers can impact performance patterns
