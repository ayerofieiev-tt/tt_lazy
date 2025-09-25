#pragma once
#include "common.hpp"

#include <cassert>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

/**
 * Shape class for tensor dimensions using SmallVector for efficient storage.
 * Uses stack allocation for small shapes (up to 8 dimensions by default).
 */
class Shape {
public:
    using value_type = uint32_t;
    using size_type = size_t;
    using iterator = SmallVector<value_type, 8>::iterator;
    using const_iterator = SmallVector<value_type, 8>::const_iterator;
    using reference = SmallVector<value_type, 8>::reference;
    using const_reference = SmallVector<value_type, 8>::const_reference;

    Shape() = default;
    Shape(std::initializer_list<value_type> dims);
    explicit Shape(const std::vector<value_type>& dims);

    template<size_t N>
    explicit Shape(const value_type (&dims)[N]);

    Shape(const Shape& other) = default;
    Shape(Shape&& other) noexcept = default;
    Shape& operator=(const Shape& other) = default;
    Shape& operator=(Shape&& other) noexcept = default;

    ~Shape() = default;

    reference operator[](size_type index) { return dims_[index]; }
    const_reference operator[](size_type index) const { return dims_[index]; }
    
    reference at(size_type index);
    const_reference at(size_type index) const;

    iterator begin() { return dims_.begin(); }
    const_iterator begin() const { return dims_.begin(); }
    const_iterator cbegin() const { return dims_.cbegin(); }
    
    iterator end() { return dims_.end(); }
    const_iterator end() const { return dims_.end(); }
    const_iterator cend() const { return dims_.cend(); }

    bool empty() const { return dims_.empty(); }
    size_type size() const { return dims_.size(); }
    size_type rank() const { return dims_.size(); }
    size_type total_elements() const;

    void push_back(const value_type& value) { dims_.push_back(value); }
    void push_back(value_type&& value) { dims_.push_back(std::move(value)); }
    
    void pop_back() { dims_.pop_back(); }
    
    void resize(size_type count) { dims_.resize(count); }
    void resize(size_type count, const value_type& value) { dims_.resize(count, value); }
    
    void clear() { dims_.clear(); }

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;

    bool is_scalar() const { return empty(); }
    bool is_vector() const { return size() == 1; }
    bool is_matrix() const { return size() == 2; }
    
    static Shape broadcast_shapes(const Shape& shape1, const Shape& shape2);
    static bool can_broadcast(const Shape& shape1, const Shape& shape2);
    
    std::vector<value_type> to_vector() const;
    std::string to_string() const;
    
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape);

private:
    SmallVector<value_type, 8> dims_;
};

template<size_t N>
Shape::Shape(const value_type (&dims)[N]) : dims_(dims, dims + N) {
    for (const auto& dim : dims_) {
        assert(dim > 0 && "Shape dimensions must be positive");
    }
}
