#include <span>
#pragma once
class Value;

class Operation : public std::enable_shared_from_this<Operation>{
    /**
     * Defines the name of an operation, what it does to inputs, and how to propagate gradients through it.
     * 
     * TODO: befriend public facing APIs for operations so we can prohibit access to forward/backward from outside.
     * We want to do this to avoid misuse, since people can pass aribtrarily sized input spans and cause runtime errors or confusion.
     */
    public:
        virtual ~Operation() = default;

        // computes output Value given inputs, returns a shared_ptr to assure that Values have identity in the computation graph
        // this method will also set the dependencies (children) of the output Value to be the inputs
        virtual std::shared_ptr<Value> forward(std::span<std::shared_ptr<Value> const> inputs) const = 0;

        // backward accumulates grads into inputs from the direct output of those inputs. This is the gradient of some final output w.r.t the output of this operation, not necessarily the immediate output
        // TODO: refactor to validate that out.operation == this
        virtual void backward(std::span<std::shared_ptr<Value> const> inputs, std::shared_ptr<const Value> out) const = 0;

        virtual std::string get_name() const = 0;

};

#define DECLARE_OPERATION_CLASS(OP_NAME) \
 class OP_NAME : public Operation { \
     public: \
         std::shared_ptr<Value> forward(std::span<std::shared_ptr<Value> const> inputs) const override; \
  \
         void backward(std::span<std::shared_ptr<Value> const> inputs, std::shared_ptr<const Value> out) const override; \
  \
         std::string get_name() const override; \
 };
DECLARE_OPERATION_CLASS(Add)
DECLARE_OPERATION_CLASS(Subtract)
DECLARE_OPERATION_CLASS(Multiply)
DECLARE_OPERATION_CLASS(Divide)
DECLARE_OPERATION_CLASS(Exp)
DECLARE_OPERATION_CLASS(Tanh)

namespace operation {
/**
 * We  set the children to be the operands involved in the operation, so taht we can trace back during backpropagation.
 *
 * WE only define this on shared pointers to make sure we don't copy around values and miss updating dependencies in the computation graph.
 */
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> tanh(const std::shared_ptr<Value> &x);
std::shared_ptr<Value> exp(const std::shared_ptr<Value> &x); // e^x


// addition operations for floats directly
std::shared_ptr<Value> operator+(float a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a, float b);
std::shared_ptr<Value> operator-(float a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a, float b);
std::shared_ptr<Value> operator*(float a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a, float b);
std::shared_ptr<Value> operator/(float a, const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a, float b);
std::shared_ptr<Value> exp(float x); // e^x
}