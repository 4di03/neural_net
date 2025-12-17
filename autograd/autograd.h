/**
 * Custom implementation of a simple neural network, just for fun and learning.
 *
 * For autograd, we approximate deirvatives using finite differences.
 */
#include <iostream>
#include <optional>
#pragma once

template <class T>
void print(const T &x)
{
    std::cout << x << '\n'; // works for anything with operator<<
}

enum class Operation
{
    Add,
    Subtract,
    Multiply,
    Divide
};
std::string to_string(const Operation &op);


std::ostream &operator<<(std::ostream &os, const Operation &op);

class Value
{
public:

    Value(float data, const std::optional<std::string>& label) : data(data), label(label) {}
    Value(float data, const std::vector<std::shared_ptr<Value>> &prev, std::optional<Operation> op) : data(data), prev(prev), op(op){}
    Value(float data, const std::vector<std::shared_ptr<Value>> &prev, std::optional<Operation> op, const std::optional<std::string>& label) : data(data), prev(prev), op(op), label(label) {}

    float get_data() const
    {
        return data;
    }

    const std::optional<std::string>& get_label() const
    {
        return label;
    }
    void set_label(const std::string& new_label)
    {
        label = new_label;
    }

    const std::vector<std::shared_ptr<Value>> &get_prev() const
    {
        return prev;
    }
    const std::optional<Operation> &get_operation() const
    {
        return op;
    }

private:
    float data; // scalar value held by this Value node
    std::vector<std::shared_ptr<Value>> prev;   // if this value is the result of an operation, store the operands
    std::optional<Operation> op = std::nullopt; // the operation that produced this value, if its not an operation, this is nullopt
    std::optional<std::string> label = std::nullopt;
};

/**
 * We  set the children to be the operands involved in the operation, so taht we can trace back during backpropagation.
 *
 * WE only define this on shared pointers to make sure we don't copy around values and miss updating dependencies in the computation graph.
 */
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);


// cout overload for shared_ptr<Value>
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Value> &v);

/**
 * Approximate derivative using (f(x + h)) - f(x)) / h for a small x
 *
 *
 * Allows use to compute derivates for general functions of from f(x), where f is any callable object
 */
std::shared_ptr<Value> make_value(float x, const std::optional<std::string>& label = std::nullopt);
