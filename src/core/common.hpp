#pragma once
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container/small_vector.hpp>

// Forward declarations
class Tensor;
class Node;
class Context;

// Fast operation type ID system
using OpTypeId = uint32_t;
using NodeId = uint32_t;

// Constants
constexpr NodeId INVALID_NODE_ID = 0;

// Use Boost's small_vector for efficient small collections
template <typename T, size_t N = 4>
using SmallVector = boost::container::small_vector<T, N>;

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

// Macro to easily define operation argument structures
#define DEFINE_OP_ARGS(OpName, ...)                         \
    struct OpName##Args : public OpArgsImpl<OpName##Args> { \
        static constexpr const char* NAME = #OpName;        \
        std::string_view op_name() const override {         \
            return NAME;                                    \
        }                                                   \
        virtual ~OpName##Args() = default;                  \
        __VA_ARGS__                                         \
    }
