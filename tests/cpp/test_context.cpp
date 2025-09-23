#include <gtest/gtest.h>
#include "common.hpp"
#include "Tensor.hpp"
#include "Node.hpp"
#include "Context.hpp"
#include "operations.hpp"


class ContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        Context::instance().clear();
    }
    
    void TearDown() override {
        Context::instance().clear();
    }
};

TEST_F(ContextTest, Singleton) {
    auto& ctx1 = Context::instance();
    auto& ctx2 = Context::instance();
    
    EXPECT_EQ(&ctx1, &ctx2);
}

TEST_F(ContextTest, NodeCreation) {
    auto& ctx = Context::instance();
    
    float data[100];
    Tensor input(data, {10, 10});
    
    SmallVector<Tensor, 2> inputs = {input};
    MatMulArgs args;
    args.transpose_a = false;
    args.transpose_b = false;
    
    auto node_id = ctx.create_node(inputs, std::move(args));
    
    EXPECT_GT(node_id, 0);
    EXPECT_EQ(ctx.size(), 1);
    
    auto* node = ctx.get_node(node_id);
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->id(), node_id);
    EXPECT_EQ(node->op_name(), "MatMul");
}

TEST_F(ContextTest, MultipleNodes) {
    auto& ctx = Context::instance();
    
    float data[100];
    Tensor input1(data, {10, 10});
    Tensor input2(data, {10, 10});
    
    auto id1 = ctx.create_node({input1}, ReLUArgs{});
    auto id2 = ctx.create_node({input2}, ReLUArgs{});
    
    EXPECT_NE(id1, id2);
    EXPECT_EQ(ctx.size(), 2);
    
    auto* node1 = ctx.get_node(id1);
    auto* node2 = ctx.get_node(id2);
    
    EXPECT_NE(node1, nullptr);
    EXPECT_NE(node2, nullptr);
    EXPECT_NE(node1, node2);
}

TEST_F(ContextTest, FindNodes) {
    auto& ctx = Context::instance();
    
    float data[100];
    Tensor input1(data, {10, 10});
    Tensor input2(data, {10, 10});
    Tensor input3(data, {10, 10});
    
    ctx.create_node({input1}, ReLUArgs{});
    MatMulArgs matmul_args;
    matmul_args.transpose_a = false;
    matmul_args.transpose_b = false;
    ctx.create_node({input2}, std::move(matmul_args));
    ctx.create_node({input3}, ReLUArgs{});
    
    auto relu_nodes = ctx.find_nodes<ReLUArgs>();
    auto matmul_nodes = ctx.find_nodes<MatMulArgs>();
    
    EXPECT_EQ(relu_nodes.size(), 2);
    EXPECT_EQ(matmul_nodes.size(), 1);
    
    for (const auto* node : relu_nodes) {
        EXPECT_EQ(node->op_name(), "ReLU");
    }
    
    for (const auto* node : matmul_nodes) {
        EXPECT_EQ(node->op_name(), "MatMul");
    }
}

TEST_F(ContextTest, Dependencies) {
    auto& ctx = Context::instance();
    
    float data[100];
    Tensor input(data, {10, 10});
    
    // Create a simple chain: input -> matmul -> relu
    auto matmul_result = matmul(input, input);
    auto relu_result = relu(matmul_result);
    
    std::vector<Tensor> outputs = {relu_result};
    auto deps = ctx.get_dependencies(outputs);
    
    EXPECT_GT(deps.size(), 0);
    
    // Should include both matmul and relu nodes
    EXPECT_GE(deps.size(), 2);
}

TEST_F(ContextTest, TopologicalSort) {
    auto& ctx = Context::instance();
    
    float data[100];
    Tensor input(data, {10, 10});
    
    auto matmul_result = matmul(input, input);
    auto relu_result = relu(matmul_result);
    
    std::vector<Tensor> outputs = {relu_result};
    auto deps = ctx.get_dependencies(outputs);
    auto exec_order = ctx.topological_sort(deps);
    
    EXPECT_GT(exec_order.size(), 0);
    
    // Should have at least matmul and relu in correct order
    EXPECT_GE(exec_order.size(), 2);
}

TEST_F(ContextTest, Clear) {
    auto& ctx = Context::instance();
    
    float data[100];
    Tensor input(data, {10, 10});
    
    ctx.create_node({input}, ReLUArgs{});
    EXPECT_EQ(ctx.size(), 1);
    
    ctx.clear();
    EXPECT_EQ(ctx.size(), 0);
}
