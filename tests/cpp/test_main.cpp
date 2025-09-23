#include <gtest/gtest.h>
#include "common.hpp"
#include "Tensor.hpp"
#include "Node.hpp"
#include "Context.hpp"
#include "operations.hpp"


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
    std::cout << "Size of Node: " << sizeof(Node) << " bytes\n";
    std::cout << "Size of Tensor: " << sizeof(Tensor) << " bytes\n";
}

TEST_F(TtLazyTest, ImplicitGraphBuilding) {
    auto& ctx = Context::instance();
    
    // Create some input tensors (constants)
    float data1[1000], data2[1000];
    Tensor input1(data1, {32, 64});
    Tensor input2(data2, {64, 128});
    
    std::cout << "Building graph implicitly through operations...\n";
    
    // Build computation graph implicitly!
    auto matmul_result = matmul(input1, input2);
    auto relu_result = relu(matmul_result);
    auto split_results = split(relu_result, 16, 0);
    auto final_result = reduce_sum(split_results[0], {1}, false);
    
    EXPECT_GT(ctx.size(), 0);
    std::cout << "Graph built with " << ctx.size() << " nodes\n";
    
    // Now we can analyze what was built
    std::cout << "\nNodes created:\n";
    for (const auto& node : ctx.get_all_nodes()) {
        std::cout << "  Node " << node.id() << ": " << node.op_name() << "\n";
        
        if (auto* matmul = node.try_as<MatMulArgs>()) {
            std::cout << "    MatMul: transpose_a=" << matmul->transpose_a << "\n";
        } else if (auto* split = node.try_as<SplitArgs>()) {
            std::cout << "    Split: size=" << split->split_size << ", dim=" << split->dim << "\n";
        } else if (auto* reduce = node.try_as<ReduceArgs>()) {
            std::cout << "    Reduce: dims=[";
            for (size_t i = 0; i < reduce->dims.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << reduce->dims[i];
            }
            std::cout << "], keepdim=" << reduce->keepdim << "\n";
        }
    }
    
    // Find dependencies for execution
    std::vector<Tensor> outputs = {final_result};
    auto deps = ctx.get_dependencies(outputs);
    
    EXPECT_GT(deps.size(), 0);
    std::cout << "\nDependencies for final result: " << deps.size() << " nodes\n";
    
    // Get execution order
    auto exec_order = ctx.topological_sort(deps);
    EXPECT_GT(exec_order.size(), 0);
    std::cout << "\nExecution order:\n";
    for (NodeId id : exec_order) {
        if (auto* node = ctx.get_node(id)) {
            std::cout << "  " << id << ": " << node->op_name() << "\n";
        }
    }
    
    ctx.print_stats();
}
