#include "Tape.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_set>

#include <spdlog/spdlog.h>

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

        // Build inputs string
        std::string inputs_str;
        for (NodeId input : op->input_nodes) {
            if (!inputs_str.empty())
                inputs_str += " ";
            inputs_str += std::to_string(input);
        }

        // Build outputs string
        std::string outputs_str;
        for (NodeId output : op->output_nodes) {
            if (!outputs_str.empty())
                outputs_str += " ";
            outputs_str += std::to_string(output);
        }

        os << "  " << i << ": Node " << op->node_id << " (op_type: " << op->op_type << ")\n";
        os << "    Inputs: " << inputs_str << "\n";
        os << "    Outputs: " << outputs_str << "\n";
    }
}

void Tape::print_tape() const {
    std::ostringstream oss;
    print_tape(oss);
    spdlog::info("{}", oss.str());
}

void Tape::build_node_map() {
    node_to_op_.clear();
    for (const auto& op : operations_) {
        node_to_op_[op->node_id] = op.get();
    }
}

// apply_mlp_fusion moved to MLPFusionPass
