#include "Tape.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>

// Tape implementation
void Tape::add_operation(std::unique_ptr<TapeOperation> op) {
    operations_.push_back(std::move(op));
    build_node_map();
}

const std::vector<std::unique_ptr<TapeOperation>>& Tape::operations() const {
    return operations_;
}

TapeOperation* Tape::find_operation(NodeId node_id) {
    auto it = node_to_op_.find(node_id);
    return it != node_to_op_.end() ? it->second : nullptr;
}

const TapeOperation* Tape::find_operation(NodeId node_id) const {
    auto it = node_to_op_.find(node_id);
    return it != node_to_op_.end() ? it->second : nullptr;
}

std::vector<NodeId> Tape::get_dependencies(NodeId node_id) const {
    const TapeOperation* op = find_operation(node_id);
    return op ? op->input_nodes : std::vector<NodeId>{};
}

void Tape::eliminate_dead_code(const std::vector<Tensor>& required_outputs) {
    std::unordered_set<NodeId> required_nodes;
    
    // Collect all required nodes by traversing backwards from outputs
    std::function<void(NodeId)> collect_required = [&](NodeId node_id) {
        if (required_nodes.count(node_id)) return;
        required_nodes.insert(node_id);
        
        const TapeOperation* op = find_operation(node_id);
        if (op) {
            for (NodeId input : op->input_nodes) {
                collect_required(input);
            }
        }
    };
    
    // Start from required outputs
    for (const auto& tensor : required_outputs) {
        if (tensor.is_lazy()) {
            collect_required(tensor.producer_node());
        }
    }
    
    // Remove operations that are not required
    operations_.erase(
        std::remove_if(operations_.begin(), operations_.end(),
            [&](const std::unique_ptr<TapeOperation>& op) {
                return required_nodes.count(op->node_id) == 0;
            }),
        operations_.end()
    );
    
    build_node_map();
}

void Tape::fuse_operations() {
    // TODO: Implement operation fusion
    // This would combine multiple operations into single kernels
}

void Tape::reorder_for_memory() {
    // TODO: Implement memory-optimized reordering
    // This would reorder operations to minimize memory usage
}

bool Tape::is_valid() const {
    // Check that all operations have valid dependencies
    for (const auto& op : operations_) {
        for (NodeId input : op->input_nodes) {
            if (!find_operation(input)) {
                return false;
            }
        }
    }
    return true;
}

void Tape::validate() const {
    if (!is_valid()) {
        throw std::runtime_error("Invalid tape: missing dependencies");
    }
}

void Tape::print_tape(std::ostream& os) const {
    os << "Tape with " << operations_.size() << " operations:\n";
    for (size_t i = 0; i < operations_.size(); ++i) {
        const auto& op = operations_[i];
        os << "  " << i << ": Node " << op->node_id 
           << " (op_type: " << op->op_type << ")\n";
        os << "    Inputs: ";
        for (NodeId input : op->input_nodes) {
            os << input << " ";
        }
        os << "\n    Outputs: ";
        for (NodeId output : op->output_nodes) {
            os << output << " ";
        }
        os << "\n";
    }
}

void Tape::build_node_map() {
    node_to_op_.clear();
    for (const auto& op : operations_) {
        node_to_op_[op->node_id] = op.get();
    }
}
