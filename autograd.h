/**
 * Custom implementation of a simple neural network, just for fun and learning.
 *
 * For autograd, we approximate deirvatives using finite differences.
 */
#include <iostream>
#include <optional>
#include <span>
#pragma once
#include "operation.h"

template <class T>
void print(const T &x)
{
    std::cout << x << '\n'; // works for anything with operator<<
}



/**
* A Value represents a scalar value in the computation graph, along with its gradient and dependencies.
*
* Each Value may be the result of an Operation applied to other Values (its "prev" nodes).
* The Value tracks:
*  - data: the scalar value
*  - grad: the gradient of some final output w.r.t this value (computed during backpropagation)
*  - prev: the input Values that were used to compute this Value (if any)
*  - op: the Operation that produced this Value (if any)
*  - label: optional human-readable label for debugging/visualization
*
* The Value class provides methods to get/set these fields, and to perform backpropagation
* to compute gradients w.r.t all input Values in the computation graph.
 */
class Value : public std::enable_shared_from_this<Value>
{
public:

    Value(float data, const std::optional<std::string>& label) : data(data), label(label) {}
    Value(float data, const std::vector<std::shared_ptr<Value>> &prev, const std::shared_ptr<const Operation> op) : data(data), prev(prev), op(op){}
    Value(float data, const std::vector<std::shared_ptr<Value>> &prev, const std::shared_ptr<const Operation> op, const std::optional<std::string>& label) : data(data), prev(prev), op(op), label(label) {}

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
    const std::shared_ptr<const Operation> &get_operation() const
    {
        return op;
    }

    float get_grad() const
    {
        return grad;
    }

    void set_grad(float new_grad)
    {
        grad = new_grad;
    }

    void add_grad(float grad_increment)
    {
        this->set_grad(this->get_grad() + grad_increment);
    }

    // propagate gradients through all dependent nodes (in topological order) to compute gradients w.r.t this value for each input Value node (modifying the grad field of each Value)
    // the gradient of this value w.r.t itself is 1.0, so a guaranteed outcome is that after calling backward on some final output Value node, that node will have grad = 1.0
    void backward();
    
private:
    float data; // scalar value held by this Value node
    
    float grad = 0.0f; 
    // this represents the gradient of some final output w.r.t this value, to be computed during backpropagation
    // the gradient tells us how much changing this value by an infinitesimal amount would change the final output at some point
    // in the computation graph of the final value, this value will be one of the nodes
    // if this is 0, it means this value has not effect on the final output


    std::vector<std::shared_ptr<Value>> prev;   // if this value is the result of an operation, store the operands
    std::shared_ptr<const Operation> op = nullptr; // the operation that produced this value, if its not an operation, this is null
    std::optional<std::string> label = std::nullopt;
};


std::ostream &operator<<(std::ostream &os, const Operation &op);



// cout overload for shared_ptr<Value>
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Value> &v);

/**
 * Approximate derivative using (f(x + h)) - f(x)) / h for a small x
 *
 *
 * Allows use to compute derivates for general functions of from f(x), where f is any callable object
 */
std::shared_ptr<Value> make_value(float x, const std::optional<std::string>& label = std::nullopt);
