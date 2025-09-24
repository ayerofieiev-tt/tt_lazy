#pragma once
#include "common.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Memory pool for efficient tensor allocation
class MemoryPool {
   public:
    static constexpr size_t DEFAULT_POOL_SIZE_MB = 1;
    static constexpr size_t BYTES_PER_MB = 1024 * 1024;

    MemoryPool(size_t initial_size = DEFAULT_POOL_SIZE_MB * BYTES_PER_MB);
    ~MemoryPool();

    // Non-copyable, movable
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) noexcept;
    MemoryPool& operator=(MemoryPool&&) noexcept;

    // Allocate memory for tensor data
    void* allocate(size_t size);

    // Deallocate memory (returns to pool)
    void deallocate(void* ptr, size_t size);

    // Get pool statistics
    size_t total_allocated() const { return total_allocated_; }
    size_t total_used() const { return total_used_; }
    size_t peak_usage() const { return peak_usage_; }

    // Clear all allocations
    void clear();

   private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<Block> blocks_;
    size_t total_allocated_ = 0;
    size_t total_used_ = 0;
    size_t peak_usage_ = 0;
    std::mutex mutex_;

    void* allocate_new_block(size_t size);
    void merge_free_blocks();
};

// Reference counting for tensor results
class TensorRef {
   public:
    TensorRef(void* data, size_t size, MemoryPool* pool);
    ~TensorRef();

    // Non-copyable, non-movable (managed by shared_ptr)
    TensorRef(const TensorRef&) = delete;
    TensorRef& operator=(const TensorRef&) = delete;
    TensorRef(TensorRef&&) = delete;
    TensorRef& operator=(TensorRef&&) = delete;

    void* data() const { return data_; }
    size_t size() const { return size_; }

    // Reference counting
    void add_ref();
    void remove_ref();
    int ref_count() const { return ref_count_; }

   private:
    void* data_;
    size_t size_;
    MemoryPool* pool_;
    std::atomic<int> ref_count_;
};

// Memory manager for the evaluation system
class MemoryManager {
   public:
    MemoryManager();
    ~MemoryManager();

    // Non-copyable, non-movable (singleton)
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;

    // Allocate tensor data with reference counting
    std::shared_ptr<TensorRef> allocate_tensor(size_t size);

    // Get memory statistics
    struct Stats {
        size_t total_allocated = 0;
        size_t total_used = 0;
        size_t peak_usage = 0;
        size_t active_tensors = 0;
        size_t memory_fragmentation = 0;
    };

    Stats get_stats() const;
    void reset_stats();

    // Memory optimization
    void garbage_collect();
    void compact_memory();

    // Global instance
    static MemoryManager& instance();

   private:
    std::unique_ptr<MemoryPool> pool_;
    std::atomic<size_t> active_tensors_;
    mutable std::mutex stats_mutex_;
    Stats stats_;

    void update_stats();
};

// RAII wrapper for tensor memory
template <typename T>
class TensorMemory {
   public:
    TensorMemory(size_t num_elements) : ref_(MemoryManager::instance().allocate_tensor(num_elements * sizeof(T))) {}

    T* data() { return static_cast<T*>(ref_->data()); }
    const T* data() const { return static_cast<const T*>(ref_->data()); }

    size_t size() const { return ref_->size() / sizeof(T); }
    size_t bytes() const { return ref_->size(); }

    bool valid() const { return ref_ != nullptr; }

   private:
    std::shared_ptr<TensorRef> ref_;
};
