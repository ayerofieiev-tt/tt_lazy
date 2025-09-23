# TT Lazy Tests

This directory contains all tests for the TT Lazy project, organized by language.

## Structure

- `cpp/` - C++ unit tests using Google Test
- `python/` - Python integration tests for the Python bindings

## Running Tests

### C++ Tests
```bash
# Build and run C++ tests
./build_with_conan.sh
```

### Python Tests
```bash
# Run Python tests (requires numpy)
cd tests/python
python3 run_tests.py
```

## Test Organization

- **C++ Tests**: Unit tests for core functionality (Tensor, Node, Context, Operations)
- **Python Tests**: Integration tests for Python bindings, including numpy interoperability
