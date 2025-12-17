/**
 * Custom implementation of a simple neural network, just for fun and learning.
 *
 * For autograd, we approximate deirvatives using finite differences.
 */
#include <iostream>
#include <optional>
#include "autograd.h"

// string overload for Operation
std::string to_string(const Operation &op)
{
    switch (op)
    {
    case Operation::Add:
        return "+";
    case Operation::Subtract:
        return "-";
    case Operation::Multiply:
        return "*";
    case Operation::Divide:
        return "/";
    default:
        return "Unknown Operation";
    }
}



std::ostream &operator<<(std::ostream &os, const Operation &op){
   os << to_string(op);
    return os;
}


/**
 * We  set the children to be the operands involved in the operation, so taht we can trace back during backpropagation.
 *
 * WE only define this on shared pointers to make sure we don't copy around values and miss updating dependencies in the computation graph.
 */
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    return std::make_shared<Value>(a->get_data() + b->get_data(), std::vector<std::shared_ptr<Value>>{a, b}, Operation::Add);
}
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    return std::make_shared<Value>(a->get_data() - b->get_data(), std::vector<std::shared_ptr<Value>>{a, b}, Operation::Subtract);
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    return std::make_shared<Value>(a->get_data() * b->get_data(), std::vector<std::shared_ptr<Value>>{a, b}, Operation::Multiply);
}
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    if (b->get_data() == 0)
    {
        throw std::runtime_error("Division by zero");
    }
    return std::make_shared<Value>(a->get_data() / b->get_data(), std::vector<std::shared_ptr<Value>>{a, b}, Operation::Divide);
}

// cout overload for shared_ptr<Value>
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Value> &v)
{
    os << "\nValue(data=" << v->get_data() << ", operation = " << (v->get_operation().has_value() ? to_string(v->get_operation().value()) : "nullopt") << ", prev=[";
    const auto &prev = v->get_prev();
    for (size_t i = 0; i < prev.size(); i++)
    {
        os << prev[i] << (i < prev.size() - 1 ? ", " : "");
    }
    os << "])\n";
    return os;
}

/**
 * Approximate derivative using (f(x + h)) - f(x)) / h for a small x
 *
 *
 * Allows use to compute derivates for general functions of from f(x), where f is any callable object
 */
std::shared_ptr<Value> make_value(float x, const std::optional<std::string>& label)
{
    return std::make_shared<Value>(x, label);
}
