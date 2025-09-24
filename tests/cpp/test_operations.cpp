#include "Context.hpp"
#include "Node.hpp"
#include "Tensor.hpp"
#include "common.hpp"
#include "operations.hpp"

#include <gtest/gtest.h>

class OperationsTest : public ::testing::Test {
   protected:
    void SetUp() override { Context::instance().clear(); }

    void TearDown() override { Context::instance().clear(); }
};

TEST_F(OperationsTest, MatMul) {
    auto& ctx = Context::instance();

    float data1[100], data2[100];
    Tensor input1(data1, {10, 10});
    Tensor input2(data2, {10, 10});

    auto result = matmul(input1, input2);

    EXPECT_EQ(ctx.size(), 1);

    auto* node = ctx.get_node(result.producer_node());
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->op_name(), "MatMul");

    const auto& args = node->as<MatMulArgs>();
    EXPECT_FALSE(args.transpose_a);
    EXPECT_FALSE(args.transpose_b);
}

TEST_F(OperationsTest, MatMulWithTranspose) {
    auto& ctx = Context::instance();

    float data1[100], data2[100];
    Tensor input1(data1, {10, 10});
    Tensor input2(data2, {10, 10});

    auto result = matmul(input1, input2, true, false);

    EXPECT_EQ(ctx.size(), 1);

    auto* node = ctx.get_node(result.producer_node());
    EXPECT_NE(node, nullptr);

    const auto& args = node->as<MatMulArgs>();
    EXPECT_TRUE(args.transpose_a);
    EXPECT_FALSE(args.transpose_b);
}

TEST_F(OperationsTest, Relu) {
    auto& ctx = Context::instance();

    float data[100];
    Tensor input(data, {10, 10});

    auto result = relu(input);

    EXPECT_EQ(ctx.size(), 1);

    auto* node = ctx.get_node(result.producer_node());
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->op_name(), "ReLU");
}

TEST_F(OperationsTest, Split) {
    auto& ctx = Context::instance();

    float data[100];
    Tensor input(data, {10, 10});

    auto results = split(input, 5, 0);

    EXPECT_EQ(ctx.size(), 1);
    EXPECT_EQ(results.size(), 2);  // Should split into 2 parts

    auto* node = ctx.get_node(results[0].producer_node());
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->op_name(), "Split");

    const auto& args = node->as<SplitArgs>();
    EXPECT_EQ(args.split_size, 5);
    EXPECT_EQ(args.dim, 0);
}

TEST_F(OperationsTest, ReduceSum) {
    auto& ctx = Context::instance();

    float data[100];
    Tensor input(data, {10, 10});

    auto result = reduce_sum(input, {1}, false);

    EXPECT_EQ(ctx.size(), 1);

    auto* node = ctx.get_node(result.producer_node());
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->op_name(), "Reduce");

    const auto& args = node->as<ReduceArgs>();
    EXPECT_EQ(args.dims.size(), 1);
    EXPECT_EQ(args.dims[0], 1);
    EXPECT_FALSE(args.keepdim);
}

TEST_F(OperationsTest, ComplexGraph) {
    auto& ctx = Context::instance();

    float data1[100], data2[100];
    Tensor input1(data1, {10, 10});
    Tensor input2(data2, {10, 10});

    // Build: input1 -> matmul -> relu -> split -> reduce_sum
    auto matmul_result = matmul(input1, input2);
    auto relu_result = relu(matmul_result);
    auto split_results = split(relu_result, 5, 0);
    auto final_result = reduce_sum(split_results[0], {1}, false);

    EXPECT_EQ(ctx.size(), 4);  // 4 operations

    // Verify the chain
    auto deps = ctx.get_dependencies({final_result});
    EXPECT_GE(deps.size(), 4);

    auto exec_order = ctx.topological_sort(deps);
    EXPECT_GE(exec_order.size(), 4);
}
