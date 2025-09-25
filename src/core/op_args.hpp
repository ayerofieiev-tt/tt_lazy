#pragma once
#include "common.hpp"
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <type_traits>
#include <cstdlib>


using OpTypeId = uint32_t;

namespace detail {
inline OpTypeId next_op_id() {
    static OpTypeId counter = 0;
    return ++counter;
}

template <typename T>
OpTypeId get_op_id() {
    static const OpTypeId id = next_op_id();
    return id;
}
}  // namespace detail

// Type erasure vtable - stores function pointers for operations
struct OpArgsVTable {
    std::string_view (*get_op_name)(const void* obj);
    void (*destroy)(void* obj);
    void (*copy_construct)(void* dest, const void* src);
    void (*move_construct)(void* dest, void* src);
    OpTypeId (*get_type_id)();
    size_t object_size;
    size_t object_alignment;
};

// Template to generate vtable for specific types
template<typename T>
constexpr OpArgsVTable make_vtable() {
    return OpArgsVTable{
        // get_op_name - requires T to have op_name() method
        [](const void* obj) -> std::string_view {
            return static_cast<const T*>(obj)->op_name();
        },
        // destroy
        [](void* obj) {
            static_cast<T*>(obj)->~T();
        },
        // copy_construct
        [](void* dest, const void* src) {
            new (dest) T(*static_cast<const T*>(src));
        },
        // move_construct
        [](void* dest, void* src) {
            new (dest) T(std::move(*static_cast<T*>(src)));
        },
        // get_type_id
        []() -> OpTypeId {
            return detail::get_op_id<T>();
        },
        sizeof(T),
        alignof(T)
    };
}

// Type-erased container for operation arguments
class OpArgs {
private:
    static constexpr size_t SMALL_BUFFER_SIZE = 64;
    static constexpr size_t SMALL_BUFFER_ALIGN = 8;
    
    // Small buffer optimization
    alignas(SMALL_BUFFER_ALIGN) char small_buffer_[SMALL_BUFFER_SIZE];
    void* data_;
    const OpArgsVTable* vtable_;
    
    bool uses_small_buffer() const {
        return data_ == static_cast<const void*>(small_buffer_);
    }
    
    void allocate_storage(size_t size, size_t alignment) {
        if (size <= SMALL_BUFFER_SIZE && alignment <= SMALL_BUFFER_ALIGN) {
            data_ = small_buffer_;
        } else {
            // Use aligned allocation for proper alignment
            void* ptr = std::aligned_alloc(alignment, size);
            if (!ptr) {
                throw std::bad_alloc();
            }
            data_ = ptr;
        }
    }
    
    void cleanup() {
        if (vtable_) {
            vtable_->destroy(data_);
            if (!uses_small_buffer()) {
                std::free(data_);
            }
        }
    }
    
public:
    OpArgs() : data_(nullptr), vtable_(nullptr) {}
    
    template<typename T>
    OpArgs(T&& value) {
        static_assert(std::is_same_v<std::decay_t<T>, T> || std::is_rvalue_reference_v<T&&>, 
                      "Use OpArgs::make<T>() for perfect forwarding");
        
        static const OpArgsVTable vtable = make_vtable<std::decay_t<T>>();
        vtable_ = &vtable;
        
        using DecayedT = std::decay_t<T>;
        allocate_storage(sizeof(DecayedT), alignof(DecayedT));
        new (data_) DecayedT(std::forward<T>(value));
    }
    
    template<typename T, typename... Args>
    static OpArgs make(Args&&... args) {
        OpArgs result;
        static const OpArgsVTable vtable = make_vtable<T>();
        result.vtable_ = &vtable;
        
        result.allocate_storage(sizeof(T), alignof(T));
        new (result.data_) T(std::forward<Args>(args)...);
        return result;
    }
    
    ~OpArgs() {
        cleanup();
    }
    
    OpArgs(const OpArgs& other) : vtable_(other.vtable_) {
        if (vtable_) {
            allocate_storage(vtable_->object_size, vtable_->object_alignment);
            vtable_->copy_construct(data_, other.data_);
        } else {
            data_ = nullptr;
        }
    }
    
    OpArgs(OpArgs&& other) noexcept 
        : vtable_(other.vtable_) {
        if (vtable_) {
            if (other.uses_small_buffer()) {
                // Move from small buffer to small buffer
                data_ = small_buffer_;
                vtable_->move_construct(data_, other.data_);
                vtable_->destroy(other.data_);
            } else {
                // Move heap allocation
                data_ = other.data_;
            }
        } else {
            data_ = nullptr;
        }
        other.data_ = nullptr;
        other.vtable_ = nullptr;
    }
    
    OpArgs& operator=(const OpArgs& other) {
        if (this != &other) {
            cleanup();
            vtable_ = other.vtable_;
            if (vtable_) {
                allocate_storage(vtable_->object_size, vtable_->object_alignment);
                vtable_->copy_construct(data_, other.data_);
            } else {
                data_ = nullptr;
            }
        }
        return *this;
    }
    
    OpArgs& operator=(OpArgs&& other) noexcept {
        if (this != &other) {
            cleanup();
            vtable_ = other.vtable_;
            if (vtable_) {
                if (other.uses_small_buffer()) {
                    // Move from small buffer to small buffer
                    data_ = small_buffer_;
                    vtable_->move_construct(data_, other.data_);
                    vtable_->destroy(other.data_);
                } else {
                    // Move heap allocation
                    data_ = other.data_;
                }
            } else {
                data_ = nullptr;
            }
            other.data_ = nullptr;
            other.vtable_ = nullptr;
        }
        return *this;
    }
    
    bool has_value() const {
        return vtable_ != nullptr;
    }
    
    std::string_view op_name() const {
        if (!vtable_) {
            throw std::runtime_error("OpArgs is empty");
        }
        return vtable_->get_op_name(data_);
    }
    
    OpTypeId type_id() const {
        if (!vtable_) {
            throw std::runtime_error("OpArgs is empty");
        }
        return vtable_->get_type_id();
    }
    
    template<typename T>
    bool is() const {
        return vtable_ && vtable_->get_type_id() == detail::get_op_id<T>();
    }
    
    template<typename T>
    T* try_cast() {
        if (!vtable_ || vtable_->get_type_id() != detail::get_op_id<T>()) {
            return nullptr;
        }
        return static_cast<T*>(data_);
    }
    
    template<typename T>
    const T* try_cast() const {
        if (!vtable_ || vtable_->get_type_id() != detail::get_op_id<T>()) {
            return nullptr;
        }
        return static_cast<const T*>(data_);
    }
    
    template<typename T>
    T& cast() {
        T* ptr = try_cast<T>();
        if (!ptr) {
            throw std::runtime_error("Invalid cast in OpArgs");
        }
        return *ptr;
    }
    
    template<typename T>
    const T& cast() const {
        const T* ptr = try_cast<T>();
        if (!ptr) {
            throw std::runtime_error("Invalid cast in OpArgs");
        }
        return *ptr;
    }
};

#define DEFINE_OP_ARGS(OpName, ...)                         \
    struct OpName##Args {                                   \
        static constexpr const char* NAME = #OpName;        \
        std::string_view op_name() const {                  \
            return NAME;                                    \
        }                                                   \
        __VA_ARGS__                                         \
    }