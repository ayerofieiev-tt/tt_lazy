#include "TapeOptimizationPass.hpp"
#include "Tape.hpp"

// Implementation of helper methods for accessing Tape internals
std::vector<std::unique_ptr<TapeOperation>>& TapeOptimizationPass::get_operations(Tape& tape) {
    return const_cast<std::vector<std::unique_ptr<TapeOperation>>&>(tape.operations());
}

void TapeOptimizationPass::rebuild_node_map(Tape& tape) {
    tape.build_node_map();
}