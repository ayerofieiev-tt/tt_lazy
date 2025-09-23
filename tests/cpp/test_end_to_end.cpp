#include <gtest/gtest.h>
#include "common.hpp"
#include "Tensor.hpp"
#include "Context.hpp"
#include "operations.hpp"
#include "tape/TapeEvaluationManager.hpp"
#include "tape/TapeExecutor.hpp"
#include "tape/TapeGenerator.hpp"
#include <chrono>
#include <random>

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
    void fill_test_data(float* data, size_t size, float value = 1.0f) {
        std::fill(data, data + size, value);
    }
    
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
                << "Data mismatch at index " << i << ": expected " << expected[i] 
                << ", got " << data[i];
        }
    }
};

TEST_F(EndToEndTest, SimpleMatMulEvaluation) {
    std::cout << "\n=== Testing Simple MatMul Evaluation ===" << std::endl;
    
    // Create input tensors with known data
    float data1[4], data2[4];  // 2x2 matrices
    fill_test_data(data1, 4, 2.0f);  // filled with 2.0
    fill_test_data(data2, 4, 3.0f);  // filled with 3.0
    
    Tensor input1(data1, {2, 2});
    Tensor input2(data2, {2, 2});
    
    // Build lazy computation graph
    auto result = matmul(input1, input2);
    
    // Verify it's lazy initially
    EXPECT_TRUE(result.is_lazy()) << "Result should be lazy before evaluation";
    EXPECT_FALSE(result.is_evaluated()) << "Result should not be evaluated yet";
    
    std::cout << "Built lazy MatMul operation" << std::endl;
    
    // Force evaluation through materialization
    auto start_time = std::chrono::high_resolution_clock::now();
    result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Materialization took: " << duration.count() << " microseconds" << std::endl;
    
    // Verify result is now materialized
    EXPECT_TRUE(result.is_evaluated()) << "Result should be materialized after evaluation";
    EXPECT_FALSE(result.is_lazy()) << "Result should not be lazy after evaluation";
    
    // Verify the computation result
    // 2x2 matrix filled with 2.0 * 2x2 matrix filled with 3.0 = 2x2 matrix filled with 12.0
    std::vector<float> expected(4, 12.0f);
    verify_tensor_data(result, expected);
    
    std::cout << "MatMul evaluation successful!" << std::endl;
}

TEST_F(EndToEndTest, ReLUActivationEvaluation) {
    std::cout << "\n=== Testing ReLU Activation Evaluation ===" << std::endl;
    
    // Create input with mixed positive and negative values
    float input_data[8] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, -3.0f};
    Tensor input(input_data, {2, 4});
    
    // Build lazy ReLU operation
    auto result = relu(input);
    
    EXPECT_TRUE(result.is_lazy()) << "Result should be lazy before evaluation";
    
    std::cout << "Built lazy ReLU operation" << std::endl;
    
    // Force evaluation
    auto start_time = std::chrono::high_resolution_clock::now();
    result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "ReLU materialization took: " << duration.count() << " microseconds" << std::endl;
    
    // Verify result
    EXPECT_TRUE(result.is_evaluated()) << "Result should be materialized after evaluation";
    
    // Expected: ReLU(x) = max(0, x)
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.5f, 0.0f};
    verify_tensor_data(result, expected);
    
    std::cout << "ReLU evaluation successful!" << std::endl;
}

TEST_F(EndToEndTest, ComplexGraphEvaluation) {
    std::cout << "\n=== Testing Complex Graph Evaluation ===" << std::endl;
    
    // Create input tensors
    float data1[16], data2[16];  // 4x4 matrices
    fill_random_data(data1, 16, -1.0f, 1.0f);
    fill_random_data(data2, 16, -1.0f, 1.0f);
    
    Tensor input1(data1, {4, 4});
    Tensor input2(data2, {4, 4});
    
    std::cout << "Created input tensors with random data" << std::endl;
    
    // Build complex computation graph: MatMul -> ReLU -> Split -> Reduce
    auto matmul_result = matmul(input1, input2);
    auto relu_result = relu(matmul_result);
    auto split_results = split(relu_result, 2, 0);  // Split into 2 parts along dim 0
    auto final_result = reduce_sum(split_results[0], {1}, false);  // Reduce along dim 1
    
    EXPECT_TRUE(final_result.is_lazy()) << "Final result should be lazy before evaluation";
    
    // Print graph structure
    auto& ctx = Context::instance();
    std::cout << "Built graph with " << ctx.size() << " nodes:" << std::endl;
    for (const auto& node : ctx.get_all_nodes()) {
        std::cout << "  Node " << node.id() << ": " << node.op_name() << std::endl;
    }
    
    // Get dependencies and execution order
    auto deps = ctx.get_dependencies({final_result});
    auto exec_order = ctx.topological_sort(deps);
    
    std::cout << "Execution order: ";
    for (NodeId id : exec_order) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    // Force evaluation of the entire graph
    auto start_time = std::chrono::high_resolution_clock::now();
    final_result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Complex graph evaluation took: " << duration.count() << " microseconds" << std::endl;
    
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
    
    std::cout << "Complex graph evaluation successful!" << std::endl;
}

TEST_F(EndToEndTest, TapeGenerationAndExecution) {
    std::cout << "\n=== Testing Tape Generation and Execution ===" << std::endl;
    
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
    
    std::cout << "Generated tape with " << tape->operations().size() << " operations:" << std::endl;
    for (const auto& op : tape->operations()) {
        std::cout << "  Operation " << op->node_id << ": type=" << op->op_type << std::endl;
    }
    
    // Test tape execution
    TapeExecutor executor;
    register_all_operations(executor);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    executor.execute_tape(*tape);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Tape execution took: " << duration.count() << " microseconds" << std::endl;
    
    // Get result from executor
    auto executed_result = executor.get_result(relu_result.producer_node());
    EXPECT_NE(executed_result, nullptr) << "Executed result should not be null";
    EXPECT_TRUE(executed_result->is_evaluated()) << "Executed result should be materialized";
    
    std::cout << "Tape generation and execution successful!" << std::endl;
}

TEST_F(EndToEndTest, EvaluationManagerIntegration) {
    std::cout << "\n=== Testing Evaluation Manager Integration ===" << std::endl;
    
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
    std::cout << "Evaluation manager evaluation took: " << duration.count() << " microseconds" << std::endl;
    
    EXPECT_NE(evaluated, nullptr) << "Evaluated result should not be null";
    EXPECT_TRUE(evaluated->is_evaluated()) << "Evaluated result should be materialized";
    
    // Test caching - second evaluation should be faster
    auto start_time2 = std::chrono::high_resolution_clock::now();
    auto cached_result = eval_manager.evaluate(relu_result);
    auto end_time2 = std::chrono::high_resolution_clock::now();
    
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2);
    std::cout << "Cached evaluation took: " << duration2.count() << " microseconds" << std::endl;
    
    EXPECT_NE(cached_result, nullptr) << "Cached result should not be null";
    EXPECT_TRUE(cached_result->is_evaluated()) << "Cached result should be materialized";
    
    // Verify results are the same
    verify_tensor_data(*evaluated, cached_result->to_vector());
    
    // Check evaluation statistics
    auto stats = eval_manager.get_stats();
    std::cout << "Evaluation stats:" << std::endl;
    std::cout << "  Cache hits: " << stats.cache_hits << std::endl;
    std::cout << "  Cache misses: " << stats.cache_misses << std::endl;
    std::cout << "  Operations executed: " << stats.operations_executed << std::endl;
    std::cout << "  Memory allocated: " << stats.memory_allocated << " bytes" << std::endl;
    
    EXPECT_GT(stats.cache_hits, 0) << "Should have cache hits";
    EXPECT_GT(stats.cache_misses, 0) << "Should have cache misses";
    EXPECT_GT(stats.operations_executed, 0) << "Should have executed operations";
    
    std::cout << "Evaluation manager integration successful!" << std::endl;
}

TEST_F(EndToEndTest, PerformanceBenchmark) {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
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
    auto split_results = split(relu_result, size/2, 0);
    auto final_result = reduce_sum(split_results[0], {1}, false);
    
    std::cout << "Built computation graph with " << Context::instance().size() << " nodes" << std::endl;
    std::cout << "Input tensor size: " << size << "x" << size << " (" << (size * size * sizeof(float)) << " bytes)" << std::endl;
    
    // Measure evaluation time
    auto start_time = std::chrono::high_resolution_clock::now();
    final_result.eval();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Total evaluation time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Evaluation time per element: " << (double)duration.count() / (size * size) << " microseconds" << std::endl;
    
    // Verify result
    EXPECT_TRUE(final_result.is_evaluated()) << "Result should be materialized";
    EXPECT_GT(final_result.total_elements(), 0) << "Result should have elements";
    
    // Test memory usage
    auto& eval_manager = tt_lazy::get_evaluation_manager();
    auto stats = eval_manager.get_stats();
    std::cout << "Memory allocated: " << stats.memory_allocated << " bytes" << std::endl;
    std::cout << "Memory per element: " << (double)stats.memory_allocated / (size * size) << " bytes" << std::endl;
    
    std::cout << "Performance benchmark completed!" << std::endl;
    
    // Clean up
    delete[] data1;
    delete[] data2;
}

TEST_F(EndToEndTest, MultipleEvaluationPaths) {
    std::cout << "\n=== Testing Multiple Evaluation Paths ===" << std::endl;
    
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
    std::cout << "ReLU evaluation took: " << duration1.count() << " microseconds" << std::endl;
    
    // MatMul should now be cached due to ReLU evaluation
    auto& eval_manager = tt_lazy::get_evaluation_manager();
    auto cached_matmul = eval_manager.evaluate(matmul_result);
    EXPECT_TRUE(cached_matmul->is_evaluated()) << "Cached MatMul should be materialized after ReLU evaluation";
    
    // Evaluate reduce - should be faster due to cached MatMul result
    auto start_time2 = std::chrono::high_resolution_clock::now();
    reduce_result.eval();
    auto end_time2 = std::chrono::high_resolution_clock::now();
    
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2);
    std::cout << "Reduce evaluation took: " << duration2.count() << " microseconds" << std::endl;
    
    EXPECT_TRUE(reduce_result.is_evaluated()) << "Reduce result should be materialized";
    
    // Both results should be materialized
    EXPECT_TRUE(relu_result.is_evaluated()) << "ReLU result should still be materialized";
    
    std::cout << "Multiple evaluation paths test successful!" << std::endl;
}
