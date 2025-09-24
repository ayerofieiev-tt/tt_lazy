#include "Context.hpp"
#include "Tensor.hpp"
#include "common.hpp"
#include "operations.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

class TensorTest : public ::testing::Test {
   protected:
    void SetUp() override { Context::instance().clear(); }

    void TearDown() override { Context::instance().clear(); }
};

TEST_F(TensorTest, BasicCreation) {
    float data[100];
    Tensor tensor(data, {10, 10});

    EXPECT_EQ(tensor.shape()[0], 10);
    EXPECT_EQ(tensor.shape()[1], 10);
    EXPECT_EQ(tensor.data_ptr(), data);
}

TEST_F(TensorTest, ShapeAccess) {
    float data[200];
    Tensor tensor(data, {5, 8, 5});

    EXPECT_EQ(tensor.rank(), 3);
    EXPECT_EQ(tensor.size(0), 5);
    EXPECT_EQ(tensor.size(1), 8);
    EXPECT_EQ(tensor.size(2), 5);
}

TEST_F(TensorTest, ProducerNode) {
    float data[50];
    Tensor tensor(data, {5, 10});

    // Initially no producer (constant tensor)
    EXPECT_EQ(tensor.producer_node(), 0);
    EXPECT_TRUE(tensor.is_constant());

    // Create a simple operation node
    auto result = relu(tensor);

    // The tensor itself doesn't change, but we can verify the result has a producer
    EXPECT_GT(result.producer_node(), 0);
}

TEST_F(TensorTest, GraphVisualization) {
    // Create some test data
    float data_a[100];
    float data_b[100];
    Tensor a(data_a, {10, 10});
    Tensor b(data_b, {10, 10});

    // Build a computation graph
    auto matmul_result = matmul(a, b);
    auto relu_result = relu(matmul_result);
    auto reduced = reduce_sum(relu_result, {1}, true);

    // Test string representation
    std::string graph_str = reduced.to_string();
    EXPECT_FALSE(graph_str.empty());

    // Should contain operation names
    EXPECT_TRUE(graph_str.find("Reduce") != std::string::npos);
    EXPECT_TRUE(graph_str.find("ReLU") != std::string::npos);
    EXPECT_TRUE(graph_str.find("MatMul") != std::string::npos);

    // Test stream operator
    std::ostringstream oss;
    oss << reduced;
    std::string stream_str = oss.str();
    EXPECT_FALSE(stream_str.empty());
    EXPECT_EQ(graph_str, stream_str);

    // Print to console for manual inspection
    spdlog::info("\n=== Graph Visualization Test ===");
    spdlog::info("Basic graph:");
    spdlog::info("{}", reduced.to_string());
}

TEST_F(TensorTest, ComplexGraphVisualization) {
    // Create a more complex graph
    float data[200];
    Tensor input(data, {4, 5});

    // Chain multiple operations
    auto split_results = split(input, 2, 0);
    EXPECT_EQ(split_results.size(), 2);

    auto first_split = split_results[0];
    auto second_split = split_results[1];

    auto matmul_result = matmul(first_split, second_split, true, false);
    auto relu_result = relu(matmul_result);
    auto final_reduce = reduce_sum(relu_result, {0, 1});

    // Test visualization
    spdlog::info("\n=== Complex Graph Visualization Test ===");
    spdlog::info("Final result graph:");
    spdlog::info("{}", final_reduce.to_string());

    spdlog::info("Split result 0 graph:");
    spdlog::info("{}", first_split.to_string());

    spdlog::info("Split result 1 graph:");
    spdlog::info("{}", second_split.to_string());

    // Verify the graph contains expected operations
    std::string graph_str = final_reduce.to_string();
    EXPECT_TRUE(graph_str.find("Reduce") != std::string::npos);
    EXPECT_TRUE(graph_str.find("ReLU") != std::string::npos);
    EXPECT_TRUE(graph_str.find("MatMul") != std::string::npos);
    EXPECT_TRUE(graph_str.find("Split") != std::string::npos);
}
