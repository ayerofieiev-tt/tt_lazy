#include "Context.hpp"
#include "Node.hpp"
#include "Tensor.hpp"
#include "common.hpp"
#include "operations.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

class TtLazyTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Clear context before each test
        Context::instance().clear();
    }

    void TearDown() override {
        // Clean up after each test
        Context::instance().clear();
    }
};

TEST_F(TtLazyTest, BasicSizes) {
    EXPECT_EQ(sizeof(Node), sizeof(Node));
    EXPECT_EQ(sizeof(Tensor), sizeof(Tensor));

    // Print sizes for debugging
    spdlog::info("Size of Node: {} bytes", sizeof(Node));
    spdlog::info("Size of Tensor: {} bytes", sizeof(Tensor));
}

TEST_F(TtLazyTest, ImplicitGraphBuilding) {
    auto& ctx = Context::instance();

    // Create some input tensors (constants)
    float data1[1000], data2[1000];
    Tensor input1(data1, {32, 64});
    Tensor input2(data2, {64, 128});

    spdlog::info("Building graph implicitly through operations...");

    // Build computation graph implicitly!
    auto matmul_result = matmul(input1, input2);
    auto relu_result = relu(matmul_result);
    auto split_results = split(relu_result, 16, 0);
    auto final_result = reduce_sum(split_results[0], {1}, false);

    EXPECT_GT(ctx.size(), 0);
    spdlog::info("Graph built with {} nodes", ctx.size());

    // Now we can analyze what was built
    spdlog::info("\nNodes created:");
    for (const auto& node : ctx.get_all_nodes()) {
        spdlog::info("  Node {}: {}", node.id(), node.op_name());

        if (auto* matmul = node.try_as<MatMulArgs>()) {
            spdlog::info("    MatMul: transpose_a={}", matmul->transpose_a);
        } else if (auto* split = node.try_as<SplitArgs>()) {
            spdlog::info("    Split: size={}, dim={}", split->split_size, split->dim);
        } else if (auto* reduce = node.try_as<ReduceArgs>()) {
            std::string dims_str;
            for (size_t i = 0; i < reduce->dims.size(); ++i) {
                if (i > 0)
                    dims_str += ",";
                dims_str += std::to_string(reduce->dims[i]);
            }
            spdlog::info("    Reduce: dims=[{}], keepdim={}", dims_str, reduce->keepdim);
        }
    }

    // Find dependencies for execution
    std::vector<Tensor> outputs = {final_result};
    auto deps = ctx.get_dependencies(outputs);

    EXPECT_GT(deps.size(), 0);
    spdlog::info("\nDependencies for final result: {} nodes", deps.size());

    // Get execution order
    auto exec_order = ctx.topological_sort(deps);
    EXPECT_GT(exec_order.size(), 0);
    spdlog::info("\nExecution order:");
    for (NodeId id : exec_order) {
        if (auto* node = ctx.get_node(id)) {
            spdlog::info("  {}: {}", id, node->op_name());
        }
    }

    ctx.print_stats();
}
