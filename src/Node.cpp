#include "Node.hpp"


NodeId Node::id() const { 
    return id_; 
}

OpTypeId Node::type_id() const { 
    return type_id_; 
}

OpArgsBase* Node::args_base() { 
    return reinterpret_cast<OpArgsBase*>(args_storage_); 
}

const OpArgsBase* Node::args_base() const { 
    return reinterpret_cast<const OpArgsBase*>(args_storage_); 
}

std::string_view Node::op_name() const { 
    return args_base()->op_name(); 
}

const SmallVector<Tensor, 4>& Node::inputs() const { 
    return inputs_; 
}

const SmallVector<NodeId, 2>& Node::output_nodes() const { 
    return output_nodes_; 
}

void Node::add_output_node(NodeId node_id) { 
    output_nodes_.push_back(node_id); 
}

