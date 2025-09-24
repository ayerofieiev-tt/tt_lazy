#include "Context.hpp"
#include "Node.hpp"
#include "Tensor.hpp"
#include "common.hpp"
#include "operations.hpp"

#include <gtest/gtest.h>

class NodeTest : public ::testing::Test {
   protected:
    void SetUp() override { Context::instance().clear(); }

    void TearDown() override { Context::instance().clear(); }
};

TEST_F(NodeTest, BasicCreation) {
    float data[100];
    Tensor input(data, {10, 10});

    SmallVector<Tensor, 2> inputs = {input};
    MatMulArgs args;
    args.transpose_a = false;
    args.transpose_b = false;

    Node node(1, inputs, std::move(args));

    EXPECT_EQ(node.id(), 1);
    EXPECT_EQ(node.type_id(), detail::get_op_id<MatMulArgs>());
    EXPECT_EQ(node.inputs().size(), 1);
    EXPECT_EQ(node.op_name(), "MatMul");
}

TEST_F(NodeTest, TypeChecking) {
    float data[100];
    Tensor input(data, {10, 10});

    SmallVector<Tensor, 2> inputs = {input};
    ReLUArgs args;

    Node node(1, inputs, std::move(args));

    EXPECT_TRUE(node.is<ReLUArgs>());
    EXPECT_FALSE(node.is<MatMulArgs>());
    EXPECT_EQ(node.type_id(), detail::get_op_id<ReLUArgs>());
}

TEST_F(NodeTest, ArgumentAccess) {
    float data[100];
    Tensor input(data, {10, 10});

    SmallVector<Tensor, 2> inputs = {input};
    MatMulArgs args;
    args.transpose_a = true;
    args.transpose_b = false;

    Node node(1, inputs, std::move(args));

    const auto& retrieved_args = node.as<MatMulArgs>();
    EXPECT_TRUE(retrieved_args.transpose_a);
    EXPECT_FALSE(retrieved_args.transpose_b);
}

TEST_F(NodeTest, TryAs) {
    float data[100];
    Tensor input(data, {10, 10});

    SmallVector<Tensor, 2> inputs = {input};
    ReLUArgs args;

    Node node(1, inputs, std::move(args));

    // Should succeed
    auto* relu_args = node.try_as<ReLUArgs>();
    EXPECT_NE(relu_args, nullptr);

    // Should fail
    auto* matmul_args = node.try_as<MatMulArgs>();
    EXPECT_EQ(matmul_args, nullptr);
}

TEST_F(NodeTest, OutputNodes) {
    float data[100];
    Tensor input(data, {10, 10});

    SmallVector<Tensor, 2> inputs = {input};
    ReLUArgs args;

    Node node(1, inputs, std::move(args));

    EXPECT_EQ(node.output_nodes().size(), 0);

    node.add_output_node(2);
    node.add_output_node(3);

    const auto& outputs = node.output_nodes();
    EXPECT_EQ(outputs.size(), 2);
    EXPECT_EQ(outputs[0], 2);
    EXPECT_EQ(outputs[1], 3);
}
