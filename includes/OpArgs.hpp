#pragma once
#include "common.hpp"

// Base interface for operation arguments
class OpArgsBase {
public:
    virtual std::string_view op_name() const = 0;
    
protected:
    ~OpArgsBase() = default;
};

// Template base for strongly-typed operation arguments
template<typename Derived>
class OpArgsImpl : public OpArgsBase {
public:
    // Static method to get the operation type ID
    static OpTypeId type_id() {
        return detail::get_op_id<Derived>();
    }
};