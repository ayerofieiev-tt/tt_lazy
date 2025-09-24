#!/usr/bin/env python3
"""
Test script for TT Lazy Python bindings
"""

import os
import sys

import numpy as np

# Add the build directory to Python path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))

try:
    import tt_lazy

    print("✓ Successfully imported tt_lazy module")
except ImportError as e:
    print(f"✗ Failed to import tt_lazy module: {e}")
    sys.exit(1)


def test_tensor_creation():
    """Test tensor creation and basic properties"""
    print("\n=== Testing Tensor Creation ===")

    # Create a constant tensor from numpy array
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    shape = [2, 2]

    try:
        tensor = tt_lazy.create_constant_tensor(data, shape)
        print(f"✓ Created constant tensor with shape {tensor.shape()}")
        print(f"  Rank: {tensor.rank()}")
        print(f"  Is constant: {tensor.is_constant()}")
        print(f"  Producer node: {tensor.producer_node()}")

        # Test shape access
        for i in range(tensor.rank()):
            print(f"  Size[{i}]: {tensor.size(i)}")

    except Exception as e:
        print(f"✗ Failed to create constant tensor: {e}")
        return False

    return True


def test_context_operations():
    """Test context operations"""
    print("\n=== Testing Context Operations ===")

    try:
        ctx = tt_lazy.Context.instance()
        print(f"✓ Got context instance, size: {ctx.size()}")

        # Clear context
        ctx.clear()
        print(f"✓ Cleared context, size: {ctx.size()}")

    except Exception as e:
        print(f"✗ Failed context operations: {e}")
        return False

    return True


def test_graph_operations():
    """Test graph operations"""
    print("\n=== Testing Graph Operations ===")

    try:
        # Create input tensors
        data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        data2 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

        tensor1 = tt_lazy.create_constant_tensor(data1, [2, 2])
        tensor2 = tt_lazy.create_constant_tensor(data2, [2, 2])

        print("✓ Created input tensors")

        # Test ReLU operation
        relu_result = tt_lazy.relu(tensor1)
        print(
            f"✓ ReLU operation successful, producer node: {relu_result.producer_node()}"
        )

        # Test matrix multiplication
        matmul_result = tt_lazy.matmul(tensor1, tensor2)
        print(
            f"✓ Matrix multiplication successful, producer node: {matmul_result.producer_node()}"
        )

        # Test split operation
        split_results = tt_lazy.split(tensor1, 1, 0)
        print(f"✓ Split operation successful, got {len(split_results)} results")

        # Test reduce operation
        reduce_result = tt_lazy.reduce(tensor1, [0], True)
        print(
            f"✓ Reduce operation successful, producer node: {reduce_result.producer_node()}"
        )

    except Exception as e:
        print(f"✗ Failed graph operations: {e}")
        return False

    return True


def test_node_inspection():
    """Test node inspection"""
    print("\n=== Testing Node Inspection ===")

    try:
        ctx = tt_lazy.Context.instance()
        print(f"Context size: {ctx.size()}")

        if ctx.size() > 0:
            # Get the first node (assuming it exists)
            node = ctx.get_node(1)  # Node IDs start from 1
            if node:
                print(f"✓ Got node {node.id()}")
                print(f"  Operation: {node.op_name()}")
                print(f"  Type ID: {node.type_id()}")
                print(f"  Input count: {node.input_count()}")
                print(f"  Output count: {node.output_count()}")
            else:
                print("No nodes found in context")

    except Exception as e:
        print(f"✗ Failed node inspection: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("TT Lazy Python Bindings Test")
    print("=" * 40)

    tests = [
        test_tensor_creation,
        test_context_operations,
        test_graph_operations,
        test_node_inspection,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
