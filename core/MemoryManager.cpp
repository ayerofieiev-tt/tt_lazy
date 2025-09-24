#include "MemoryManager.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include <spdlog/spdlog.h>

// MemoryPool implementation
MemoryPool::MemoryPool(size_t initial_size) {
    // Pre-allocate initial block
    if (initial_size > 0) {
        allocate_new_block(initial_size);
    }
}

MemoryPool::~MemoryPool() {
    clear();
}

MemoryPool::MemoryPool(MemoryPool&& other) noexcept
    : blocks_(std::move(other.blocks_)),
      total_allocated_(other.total_allocated_),
      total_used_(other.total_used_),
      peak_usage_(other.peak_usage_) {
    other.total_allocated_ = 0;
    other.total_used_ = 0;
    other.peak_usage_ = 0;
}

MemoryPool& MemoryPool::operator=(MemoryPool&& other) noexcept {
    if (this != &other) {
        clear();
        blocks_ = std::move(other.blocks_);
        total_allocated_ = other.total_allocated_;
        total_used_ = other.total_used_;
        peak_usage_ = other.peak_usage_;

        other.total_allocated_ = 0;
        other.total_used_ = 0;
        other.peak_usage_ = 0;
    }
    return *this;
}

void* MemoryPool::allocate(size_t size) {
    std::scoped_lock<std::mutex> lock(mutex_);

    // Find a suitable free block
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            total_used_ += size;
            peak_usage_ = std::max(peak_usage_, total_used_);
            return block.ptr;
        }
    }

    // No suitable block found, allocate new one
    return allocate_new_block(size);
}

void MemoryPool::deallocate(void* ptr, size_t size) {
    std::scoped_lock<std::mutex> lock(mutex_);

    // Find the block and mark it as free
    for (auto& block : blocks_) {
        if (block.ptr == ptr && block.in_use) {
            block.in_use = false;
            total_used_ -= size;
            merge_free_blocks();
            return;
        }
    }

    // Block not found - this shouldn't happen
    spdlog::warn("Warning: Attempting to deallocate unknown memory block");
}

void MemoryPool::clear() {
    std::scoped_lock<std::mutex> lock(mutex_);

    for (auto& block : blocks_) {
        if (block.ptr) {
            free(
                block
                    .ptr);  // NOLINT(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory) - Memory pool implementation
        }
    }

    blocks_.clear();
    total_allocated_ = 0;
    total_used_ = 0;
    peak_usage_ = 0;
}

void* MemoryPool::allocate_new_block(size_t size) {
    void* ptr = malloc(
        size);  // NOLINT(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory) - Memory pool implementation
    if (!ptr) {
        throw std::bad_alloc();
    }

    blocks_.push_back({ptr, size, true});
    total_allocated_ += size;
    total_used_ += size;
    peak_usage_ = std::max(peak_usage_, total_used_);

    return ptr;
}

void MemoryPool::merge_free_blocks() {
    // Sort blocks by pointer address
    std::sort(blocks_.begin(), blocks_.end(), [](const Block& a, const Block& b) { return a.ptr < b.ptr; });

    // Merge adjacent free blocks
    for (size_t i = 0; i < blocks_.size() - 1;) {
        Block& current = blocks_[i];
        Block& next = blocks_[i + 1];

        if (!current.in_use && !next.in_use) {
            // Check if blocks are adjacent
            char* current_end = static_cast<char*>(current.ptr) + current.size;
            if (current_end == next.ptr) {
                // Merge blocks
                current.size += next.size;
                blocks_.erase(blocks_.begin() + static_cast<ptrdiff_t>(i + 1));
                continue;
            }
        }
        ++i;
    }
}

// TensorRef implementation
TensorRef::TensorRef(void* data, size_t size, MemoryPool* pool) : data_(data), size_(size), pool_(pool), ref_count_(1) {
}

TensorRef::~TensorRef() {
    if (pool_) {
        pool_->deallocate(data_, size_);
    }
}

void TensorRef::add_ref() {
    ref_count_.fetch_add(1);
}

void TensorRef::remove_ref() {
    int count = ref_count_.fetch_sub(1);
    if (count == 1) {
        delete this;
    }
}

// MemoryManager implementation
MemoryManager::MemoryManager() : pool_(std::make_unique<MemoryPool>()), active_tensors_(0) {
    reset_stats();
}

MemoryManager::~MemoryManager() {
    garbage_collect();
}

std::shared_ptr<TensorRef> MemoryManager::allocate_tensor(size_t size) {
    void* data = pool_->allocate(size);
    if (!data) {
        throw std::bad_alloc();
    }

    active_tensors_.fetch_add(1);
    update_stats();

    return std::shared_ptr<TensorRef>(new TensorRef(data, size, pool_.get()), [this](TensorRef* ref) {
        active_tensors_.fetch_sub(1);
        delete ref;  // NOLINT(cppcoreguidelines-owning-memory) - Custom deleter for shared_ptr
        update_stats();
    });
}

MemoryManager::Stats MemoryManager::get_stats() const {
    std::scoped_lock<std::mutex> lock(stats_mutex_);
    return stats_;
}

void MemoryManager::reset_stats() {
    std::scoped_lock<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
    update_stats();
}

void MemoryManager::garbage_collect() {
    // In a more sophisticated implementation, this would:
    // 1. Identify unreferenced tensors
    // 2. Deallocate their memory
    // 3. Compact the memory pool

    // For now, just update statistics
    update_stats();
}

void MemoryManager::compact_memory() {
    // In a more sophisticated implementation, this would:
    // 1. Move allocated blocks to reduce fragmentation
    // 2. Merge free blocks
    // 3. Update all tensor references

    // For now, just trigger garbage collection
    garbage_collect();
}

void MemoryManager::update_stats() {
    std::scoped_lock<std::mutex> lock(stats_mutex_);
    stats_.total_allocated = pool_->total_allocated();
    stats_.total_used = pool_->total_used();
    stats_.peak_usage = pool_->peak_usage();
    stats_.active_tensors = active_tensors_.load();

    // Calculate fragmentation (simplified)
    stats_.memory_fragmentation = stats_.total_allocated - stats_.total_used;
}

// Global instance - use function-local static for safe initialization
MemoryManager& MemoryManager::instance() {
    static MemoryManager g_memory_manager;
    return g_memory_manager;
}
