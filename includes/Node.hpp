#pragma once
#include "common.hpp"
#include "Tensor.hpp"
#include "OpArgs.hpp"

// Graph node with intrusive storage
class Node {
public:
    template<typename ArgsT>
    Node(NodeId id, const SmallVector<Tensor, 2>& inputs, ArgsT&& args) 
        : id_(id), type_id_(detail::get_op_id<std::decay_t<ArgsT>>()), inputs_(inputs), output_nodes_() {
        static_assert(sizeof(ArgsT) <= sizeof(args_storage_), "Args too large for inline storage");
        new (args_storage_) std::decay_t<ArgsT>(std::forward<ArgsT>(args));
    }
    
    NodeId id() const;
    OpTypeId type_id() const;
    
    template<typename T>
    bool is() const { 
        return type_id_ == detail::get_op_id<T>(); 
    }
    
    template<typename T>
    const T& as() const { 
        assert(is<T>());
        return *reinterpret_cast<const T*>(args_storage_); 
    }
    
    template<typename T>
    T& as() { 
        assert(is<T>());
        return *reinterpret_cast<T*>(args_storage_); 
    }
    
    template<typename T>
    const T* try_as() const { 
        return is<T>() ? reinterpret_cast<const T*>(args_storage_) : nullptr; 
    }
    
    template<typename T>
    T* try_as() { 
        return is<T>() ? reinterpret_cast<T*>(args_storage_) : nullptr; 
    }
    
    OpArgsBase* args_base();
    const OpArgsBase* args_base() const;
    
    std::string_view op_name() const;
    
    const SmallVector<Tensor, 2>& inputs() const;
    const SmallVector<NodeId, 2>& output_nodes() const;
    
    void add_output_node(NodeId node_id);

private:
    NodeId id_;
    OpTypeId type_id_;
    SmallVector<Tensor, 2> inputs_;
    SmallVector<NodeId, 2> output_nodes_;
    alignas(std::max_align_t) char args_storage_[256];
};

