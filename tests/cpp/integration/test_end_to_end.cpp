#include "Context.hpp"
#include "TapeEvaluationManager.hpp"
#include "TapeExecutor.hpp"
#include "TapeGenerator.hpp"
#include "Tensor.hpp"
#include "common.hpp"
#include "operations.hpp"

#include <chrono>
#include <random>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

class EndToEndTest : public ::testing::Test {
   protected:
    void SetUp() override {
        Context::instance().clear();
        tt_lazy::get_evaluation_manager().clear_cache();
    }

    void TearDown() override {
        Context::instance().clear();
        tt_lazy::get_evaluation_manager().clear_cache();
    }

    // Helper to create test data arrays
    void fill_test_data(float* data, size_t size, float value = 1.0f) { std::fill(data, data + size, value); }

    // Helper to create random test data
    void fill_random_data(float* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);

        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    }

    // Helper to verify tensor data matches expected values
    void verify_tensor_data(const Tensor& tensor, const std::vector<float>& expected, float tolerance = 1e-6f) {
        ASSERT_TRUE(tensor.is_evaluated()) << "Tensor should be evaluated for data verification";
        ASSERT_EQ(tensor.total_elements(), expected.size()) << "Tensor size mismatch";

        const float* data = tensor.const_data_ptr();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(data[i], expected[i], tolerance)
                << "Data mismatch at index " << i << ": expected " << expected[i] << ", got " << data[i];
        }
    }
};

TEST_F(EndToEndTest, SimpleMatMulEvaluation) {
    spdlog::info("\n=== Testing Simple MatMul Evaluation ===");

    // Create input tensors with known data
    float data1[4], data2[4];        // 2x2 matrices
    fill_test_data(data1, 4, 2.0f);  // filled with 2.0
    fill_test_data(data2, 4, 3.0f);  // filled with 3.0

    Tensor input1(data1, {2, 2});
    Tensor input2(data2, {2, 2});

    // Build lazy computation graph
    auto result = matmul(input1, input2);

    // Verify it's lazy initially
    EXPECT_TRUE(result.is_lazy()) << "Result should be lazy before evaluation";
    EXPECT_FALSE(result.is_evaluated()) << "Result should not be evaluated yet";

    spdlog::info("Built lazy MatMul operation");

    // Force evaluation through materialization
    auto start_time = std::chrono::high_resolution_clock::now();
    result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    spdlog::info("Materialization took: {} microseconds", duration.count());

    // Verify result is now materialized
    EXPECT_TRUE(result.is_evaluated()) << "Result should be materialized after evaluation";
    EXPECT_FALSE(result.is_lazy()) << "Result should not be lazy after evaluation";

    // Verify the computation result
    // 2x2 matrix filled with 2.0 * 2x2 matrix filled with 3.0 = 2x2 matrix filled with 12.0
    std::vector<float> expected(4, 12.0f);
    verify_tensor_data(result, expected);

    spdlog::info("MatMul evaluation successful!");
}

TEST_F(EndToEndTest, ReLUActivationEvaluation) {
    spdlog::info("\n=== Testing ReLU Activation Evaluation ===");

    // Create input with mixed positive and negative values
    float input_data[8] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, -3.0f};
    Tensor input(input_data, {2, 4});

    // Build lazy ReLU operation
    auto result = relu(input);

    EXPECT_TRUE(result.is_lazy()) << "Result should be lazy before evaluation";

    spdlog::info("Built lazy ReLU operation");

    // Force evaluation
    auto start_time = std::chrono::high_resolution_clock::now();
    result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    spdlog::info("ReLU materialization took: {} microseconds", duration.count());

    // Verify result
    EXPECT_TRUE(result.is_evaluated()) << "Result should be materialized after evaluation";

    // Expected: ReLU(x) = max(0, x)
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.5f, 0.0f};
    verify_tensor_data(result, expected);

    spdlog::info("ReLU evaluation successful!");
}

TEST_F(EndToEndTest, ComplexGraphEvaluation) {
    spdlog::info("\n=== Testing Complex Graph Evaluation ===");

    // Create input tensors
    float data1[16], data2[16];  // 4x4 matrices
    fill_random_data(data1, 16, -1.0f, 1.0f);
    fill_random_data(data2, 16, -1.0f, 1.0f);

    Tensor input1(data1, {4, 4});
    Tensor input2(data2, {4, 4});

    spdlog::info("Created input tensors with random data");

    // Build complex computation graph: MatMul -> ReLU -> Split -> Reduce
    auto matmul_result = matmul(input1, input2);
    auto relu_result = relu(matmul_result);
    auto split_results = split(relu_result, 2, 0);                 // Split into 2 parts along dim 0
    auto final_result = reduce_sum(split_results[0], {1}, false);  // Reduce along dim 1

    EXPECT_TRUE(final_result.is_lazy()) << "Final result should be lazy before evaluation";

    // Print graph structure
    auto& ctx = Context::instance();
    spdlog::info("Built graph with {} nodes:", ctx.size());
    for (const auto& node : ctx.get_all_nodes()) {
        spdlog::info("  Node {}: {}", node.id(), node.op_name());
    }

    // Get dependencies and execution order
    auto deps = ctx.get_dependencies({final_result});
    auto exec_order = ctx.topological_sort(deps);

    std::string exec_order_str;
    for (NodeId id : exec_order) {
        if (!exec_order_str.empty())
            exec_order_str += " ";
        exec_order_str += std::to_string(id);
    }
    spdlog::info("Execution order: {}", exec_order_str);

    // Force evaluation of the entire graph
    auto start_time = std::chrono::high_resolution_clock::now();
    final_result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    spdlog::info("Complex graph evaluation took: {} microseconds", duration.count());

    // Verify final result is materialized
    EXPECT_TRUE(final_result.is_evaluated()) << "Final result should be materialized";

    // Verify intermediate results are also materialized (due to caching)
    // Note: The original lazy tensors are still lazy, but the evaluation manager
    // has cached materialized versions of them
    auto& eval_manager = tt_lazy::get_evaluation_manager();
    auto cached_matmul = eval_manager.evaluate(matmul_result);
    auto cached_relu = eval_manager.evaluate(relu_result);
    auto cached_split = eval_manager.evaluate(split_results[0]);

    EXPECT_TRUE(cached_matmul->is_evaluated()) << "Cached MatMul result should be materialized";
    EXPECT_TRUE(cached_relu->is_evaluated()) << "Cached ReLU result should be materialized";
    EXPECT_TRUE(cached_split->is_evaluated()) << "Cached Split result should be materialized";

    spdlog::info("Complex graph evaluation successful!");
}

TEST_F(EndToEndTest, TapeGenerationAndExecution) {
    spdlog::info("\n=== Testing Tape Generation and Execution ===");

    // Create a simple computation graph
    float data1[9], data2[9];  // 3x3 matrices
    fill_test_data(data1, 9, 1.0f);
    fill_test_data(data2, 9, 2.0f);

    Tensor input1(data1, {3, 3});
    Tensor input2(data2, {3, 3});

    auto result = matmul(input1, input2);
    auto relu_result = relu(result);

    EXPECT_TRUE(relu_result.is_lazy()) << "Result should be lazy before tape generation";

    // Test tape generation directly
    TapeGenerator generator;
    auto tape = generator.generate_tape(relu_result);

    EXPECT_NE(tape, nullptr) << "Tape should be generated successfully";
    EXPECT_GT(tape->operations().size(), 0) << "Tape should contain operations";

    spdlog::info("Generated tape with  {} operations:", tape->operations().size());
    for (const auto& op : tape->operations()) {
        spdlog::info("  Operation {}: type={}", op->node_id, op->op_type);
    }

    // Test tape execution
    TapeExecutor executor;
    register_all_operations(executor);

    auto start_time = std::chrono::high_resolution_clock::now();
    executor.execute_tape(*tape);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    spdlog::info("Tape execution took:  {} microseconds", duration.count());

    // Get result from executor
    auto executed_result = executor.get_result(relu_result.producer_node());
    EXPECT_NE(executed_result, nullptr) << "Executed result should not be null";
    EXPECT_TRUE(executed_result->is_evaluated()) << "Executed result should be materialized";

    spdlog::info("Tape generation and execution successful!");
}

TEST_F(EndToEndTest, EvaluationManagerIntegration) {
    spdlog::info("\n=== Testing Evaluation Manager Integration ===");

    auto& eval_manager = tt_lazy::get_evaluation_manager();

    // Create computation graph
    float data1[4], data2[4];  // 2x2 matrices
    fill_test_data(data1, 4, 1.0f);
    fill_test_data(data2, 4, 2.0f);

    Tensor input1(data1, {2, 2});
    Tensor input2(data2, {2, 2});

    auto result = matmul(input1, input2);
    auto relu_result = relu(result);

    // Test evaluation through evaluation manager
    auto start_time = std::chrono::high_resolution_clock::now();
    auto evaluated = eval_manager.evaluate(relu_result);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    spdlog::info("Evaluation manager evaluation took:  {} microseconds", duration.count());

    EXPECT_NE(evaluated, nullptr) << "Evaluated result should not be null";
    EXPECT_TRUE(evaluated->is_evaluated()) << "Evaluated result should be materialized";

    // Test caching - second evaluation should be faster
    auto start_time2 = std::chrono::high_resolution_clock::now();
    auto cached_result = eval_manager.evaluate(relu_result);
    auto end_time2 = std::chrono::high_resolution_clock::now();

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2);
    spdlog::info("Cached evaluation took:  {} microseconds", duration2.count());

    EXPECT_NE(cached_result, nullptr) << "Cached result should not be null";
    EXPECT_TRUE(cached_result->is_evaluated()) << "Cached result should be materialized";

    // Verify results are the same
    verify_tensor_data(*evaluated, cached_result->to_vector());

    // Check evaluation statistics
    auto stats = eval_manager.get_stats();
    spdlog::info("Evaluation stats:");
    spdlog::info("  Cache hits:  {}", stats.cache_hits);
    spdlog::info("  Cache misses:  {}", stats.cache_misses);
    spdlog::info("  Operations executed:  {}", stats.operations_executed);
    spdlog::info("  Memory allocated: {} bytes", stats.memory_allocated);

    EXPECT_GT(stats.cache_hits, 0) << "Should have cache hits";
    EXPECT_GT(stats.cache_misses, 0) << "Should have cache misses";
    EXPECT_GT(stats.operations_executed, 0) << "Should have executed operations";

    spdlog::info("Evaluation manager integration successful!");
}

TEST_F(EndToEndTest, PerformanceBenchmark) {
    spdlog::info("\n=== Performance Benchmark ===");

    // Test with larger tensors for performance measurement
    const size_t size = 64;  // 64x64 matrices
    const size_t total_size = size * size;
    float* data1 = new float[total_size];
    float* data2 = new float[total_size];
    fill_random_data(data1, total_size, -1.0f, 1.0f);
    fill_random_data(data2, total_size, -1.0f, 1.0f);

    Tensor input1(data1, {size, size});
    Tensor input2(data2, {size, size});

    // Build computation graph
    auto matmul_result = matmul(input1, input2);
    auto relu_result = relu(matmul_result);
    auto split_results = split(relu_result, size / 2, 0);
    auto final_result = reduce_sum(split_results[0], {1}, false);

    spdlog::info("Built computation graph with {} nodes", Context::instance().size());
    spdlog::info("Input tensor size: {}x{} ({} bytes)", size, size, (size * size * sizeof(float)));

    // Measure evaluation time
    auto start_time = std::chrono::high_resolution_clock::now();
    final_result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    spdlog::info("Total evaluation time: {} microseconds", duration.count());
    spdlog::info("Evaluation time per element: {} microseconds", static_cast<double>(duration.count()) / (size * size));

    // Verify result
    EXPECT_TRUE(final_result.is_evaluated()) << "Result should be materialized";
    EXPECT_GT(final_result.total_elements(), 0) << "Result should have elements";

    // Test memory usage
    auto& eval_manager = tt_lazy::get_evaluation_manager();
    auto stats = eval_manager.get_stats();
    spdlog::info("Memory allocated: {} bytes", stats.memory_allocated);
    spdlog::info("Memory per element: {} bytes", static_cast<double>(stats.memory_allocated) / (size * size));

    spdlog::info("Performance benchmark completed!");

    // Clean up
    delete[] data1;
    delete[] data2;
}

TEST_F(EndToEndTest, MultipleEvaluationPaths) {
    spdlog::info("\n=== Testing Multiple Evaluation Paths ===");

    // Create a graph where multiple operations depend on the same intermediate result
    float data1[4], data2[4];  // 2x2 matrices
    fill_test_data(data1, 4, 1.0f);
    fill_test_data(data2, 4, 2.0f);

    Tensor input1(data1, {2, 2});
    Tensor input2(data2, {2, 2});

    // Build graph: input1, input2 -> matmul -> [relu, reduce_sum]
    auto matmul_result = matmul(input1, input2);
    auto relu_result = relu(matmul_result);
    auto reduce_result = reduce_sum(matmul_result, {1}, false);

    EXPECT_TRUE(relu_result.is_lazy()) << "ReLU result should be lazy";
    EXPECT_TRUE(reduce_result.is_lazy()) << "Reduce result should be lazy";

    // Evaluate ReLU first
    auto start_time = std::chrono::high_resolution_clock::now();
    relu_result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    spdlog::info("ReLU evaluation took: {} microseconds", duration1.count());

    // MatMul should now be cached due to ReLU evaluation
    auto& eval_manager = tt_lazy::get_evaluation_manager();
    auto cached_matmul = eval_manager.evaluate(matmul_result);
    EXPECT_TRUE(cached_matmul->is_evaluated()) << "Cached MatMul should be materialized after ReLU evaluation";

    // Evaluate reduce - should be faster due to cached MatMul result
    auto start_time2 = std::chrono::high_resolution_clock::now();
    reduce_result.eval();
    auto end_time2 = std::chrono::high_resolution_clock::now();

    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2);
    spdlog::info("Reduce evaluation took: {} microseconds", duration2.count());

    EXPECT_TRUE(reduce_result.is_evaluated()) << "Reduce result should be materialized";

    // Both results should be materialized
    EXPECT_TRUE(relu_result.is_evaluated()) << "ReLU result should still be materialized";

    spdlog::info("Multiple evaluation paths test successful!");
}
