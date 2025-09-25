#include "shape.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

Shape::Shape(std::initializer_list<value_type> dims) : dims_(dims) {
    for (const auto& dim : dims_) {
        assert(dim > 0 && "Shape dimensions must be positive");
    }
}

Shape::Shape(const std::vector<value_type>& dims) : dims_(dims.begin(), dims.end()) {
    for (const auto& dim : dims_) {
        assert(dim > 0 && "Shape dimensions must be positive");
    }
}

Shape::reference Shape::at(size_type index) {
    if (index >= size()) {
        throw std::out_of_range("Shape index out of range");
    }
    return dims_[index];
}

Shape::const_reference Shape::at(size_type index) const {
    if (index >= size()) {
        throw std::out_of_range("Shape index out of range");
    }
    return dims_[index];
}

Shape::size_type Shape::total_elements() const {
    if (empty()) {
        return 0;
    }
    return std::accumulate(dims_.begin(), dims_.end(), size_type(1), std::multiplies<size_type>());
}

bool Shape::operator==(const Shape& other) const {
    return dims_ == other.dims_;
}

bool Shape::operator!=(const Shape& other) const {
    return !(*this == other);
}

Shape Shape::broadcast_shapes(const Shape& shape1, const Shape& shape2) {
    if (!can_broadcast(shape1, shape2)) {
        throw std::runtime_error("Shapes cannot be broadcast together");
    }
    
    const size_type max_rank = std::max(shape1.rank(), shape2.rank());
    Shape result;
    result.dims_.reserve(max_rank);
    
    const size_type pad1 = max_rank - shape1.rank();
    const size_type pad2 = max_rank - shape2.rank();
    
    for (size_type i = 0; i < max_rank; ++i) {
        value_type dim1 = (i < pad1) ? 1 : shape1[i - pad1];
        value_type dim2 = (i < pad2) ? 1 : shape2[i - pad2];
        
        if (dim1 == 1) {
            result.dims_.push_back(dim2);
        } else if (dim2 == 1) {
            result.dims_.push_back(dim1);
        } else if (dim1 == dim2) {
            result.dims_.push_back(dim1);
        } else {
            throw std::runtime_error("Incompatible dimensions for broadcasting");
        }
    }
    
    return result;
}

bool Shape::can_broadcast(const Shape& shape1, const Shape& shape2) {
    if (shape1.empty() || shape2.empty()) {
        return true; // Empty shapes can broadcast
    }
    
    const size_type max_rank = std::max(shape1.rank(), shape2.rank());
    const size_type pad1 = max_rank - shape1.rank();
    const size_type pad2 = max_rank - shape2.rank();
    
    for (size_type i = 0; i < max_rank; ++i) {
        value_type dim1 = (i < pad1) ? 1 : shape1[i - pad1];
        value_type dim2 = (i < pad2) ? 1 : shape2[i - pad2];
        
        if (dim1 != 1 && dim2 != 1 && dim1 != dim2) {
            return false;
        }
    }
    
    return true;
}

std::vector<Shape::value_type> Shape::to_vector() const {
    return std::vector<value_type>(dims_.begin(), dims_.end());
}

std::string Shape::to_string() const {
    if (empty()) {
        return "[]";
    }
    
    std::ostringstream oss;
    oss << "[";
    for (size_type i = 0; i < size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dims_[i];
    }
    oss << "]";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << shape.to_string();
    return os;
}
