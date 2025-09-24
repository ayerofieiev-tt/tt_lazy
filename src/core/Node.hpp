#pragma once
#include "OpArgs.hpp"
#include "Tensor.hpp"
#include "common.hpp"

// Graph node with intrusive storage
class Node {
   public:
    template <typename ArgsT>
    Node(NodeId id, const SmallVector<Tensor, 2>& inputs, ArgsT&& args)
        : id_(id), type_id_(detail::get_op_id<std::decay_t<ArgsT>>()), output_nodes_(), args_storage_{} {
        static_assert(sizeof(ArgsT) <= sizeof(args_storage_), "Args too large for inline storage");

        // Copy inputs to our larger container
        for (const auto& input : inputs) {
            inputs_.push_back(input);
        }

        new (args_storage_) std::decay_t<ArgsT>(std::forward<ArgsT>(args));
    }

    // Constructor for variable number of inputs
    template <typename ArgsT, size_t N>
    Node(NodeId id, const SmallVector<Tensor, N>& inputs, ArgsT&& args)
        : id_(id), type_id_(detail::get_op_id<std::decay_t<ArgsT>>()), output_nodes_(), args_storage_{} {
        static_assert(sizeof(ArgsT) <= sizeof(args_storage_), "Args too large for inline storage");

        // Store inputs in the fixed-size container (extend if needed)
        for (size_t i = 0; i < inputs.size() && i < inputs_.max_size(); ++i) {
            inputs_.push_back(inputs[i]);
        }

        new (args_storage_) std::decay_t<ArgsT>(std::forward<ArgsT>(args));
    }

    NodeId id() const;
    OpTypeId type_id() const;

    template <typename T>
    bool is() const {
        return type_id_ == detail::get_op_id<T>();
    }

    template <typename T>
    const T& as() const {
        assert(is<T>());
        return *reinterpret_cast<const T*>(
            args_storage_);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - Type erasure with proper type checking
    }

    template <typename T>
    T& as() {
        assert(is<T>());
        return *reinterpret_cast<T*>(
            args_storage_);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - Type erasure with proper type checking
    }

    template <typename T>
    const T* try_as() const {
        return is<T>()
                   ? reinterpret_cast<const T*>(args_storage_)
                   : nullptr;  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - Type erasure with proper type checking
    }

    template <typename T>
    T* try_as() {
        return is<T>()
                   ? reinterpret_cast<T*>(args_storage_)
                   : nullptr;  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - Type erasure with proper type checking
    }

    OpArgsBase* args_base();
    const OpArgsBase* args_base() const;

    std::string_view op_name() const;

    const SmallVector<Tensor, 4>& inputs() const;
    const SmallVector<NodeId, 2>& output_nodes() const;

    void add_output_node(NodeId node_id);

   private:
    NodeId id_;
    OpTypeId type_id_;
    SmallVector<Tensor, 4> inputs_;
    SmallVector<NodeId, 2> output_nodes_;
    static constexpr size_t ARGS_STORAGE_SIZE = 256;
    alignas(std::max_align_t) char args_storage_
        [ARGS_STORAGE_SIZE];  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Type erasure storage requires C-style array
};
