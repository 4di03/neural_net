/**
 * Custom implementation of automatic differentiation on scalar-valued functions, just for fun and learning.
 */
#include <iostream>
#include "operation.h"
#include "autograd.h"


// Implementations of Operation subclasses

std::shared_ptr<Value> Add::forward(std::span<const std::shared_ptr<Value>> inputs) const {
    if (inputs.size() != 2) {
        throw std::runtime_error("Add operation requires exactly two inputs");
    }
    float result = inputs[0]->get_data() + inputs[1]->get_data();
    return std::make_shared<Value>(result, std::vector<std::shared_ptr<Value>>(inputs.begin(), inputs.end()), shared_from_this());
}
void Add::backward(std::span<const std::shared_ptr<Value>> inputs, std::shared_ptr<const Value> out) const {
    if (inputs.size() != 2) {
        throw std::runtime_error("Add operation requires exactly two inputs");
    }
    auto out_grad = out->get_grad();
    // we add it since gradient contributions for subfunctions of x add up (linearity of differentiation)
    // Intuition: https://math.stackexchange.com/q/1327030
    inputs[0]->add_grad(out_grad);
    inputs[1]->add_grad(out_grad);
}
std::string Add::get_name() const {
    return "+";
}


std::shared_ptr<Value> Subtract::forward(std::span<const std::shared_ptr<Value>> inputs) const {
    if (inputs.size() != 2) {
        throw std::runtime_error("Subtract operation requires exactly two inputs");
    }
    float result = inputs[0]->get_data() - inputs[1]->get_data();
    return std::make_shared<Value>(result, std::vector<std::shared_ptr<Value>>(inputs.begin(), inputs.end()), shared_from_this());
}

void Subtract::backward(std::span<const std::shared_ptr<Value>> inputs, std::shared_ptr<const Value> out) const {
    if (inputs.size() != 2) {
        throw std::runtime_error("Subtract operation requires exactly two inputs");
    }
    auto out_grad = out->get_grad();
    inputs[0]->add_grad(out_grad);
    inputs[1]->add_grad(-1 * out_grad); // since its inputs[0] - inputs[1]
}

std::string Subtract::get_name() const {
    return "-";
}




std::shared_ptr<Value> Multiply::forward(std::span<const std::shared_ptr<Value>> inputs) const {
    if (inputs.size() != 2) {
        throw std::runtime_error("Multiply operation requires exactly two inputs");
    }
    float result = inputs[0]->get_data() * inputs[1]->get_data();
    return std::make_shared<Value>(result, std::vector<std::shared_ptr<Value>>(inputs.begin(), inputs.end()), shared_from_this());
}

void Multiply::backward(std::span<const std::shared_ptr<Value>> inputs, std::shared_ptr<const Value> out) const {
    if (inputs.size() != 2) {
        throw std::runtime_error("Multiply operation requires exactly two inputs");
    }
    // using the product rule
    inputs[0]->add_grad(inputs[1]->get_data() * out->get_grad());
    inputs[1]->add_grad(inputs[0]->get_data() * out->get_grad());
}

std::string Multiply::get_name() const {
    return "*";
}


std::shared_ptr<Value> Divide::forward(std::span<const std::shared_ptr<Value>> inputs) const {
    if (inputs.size() != 2) {
        throw std::runtime_error("Divide operation requires exactly two inputs");
    }
    if (inputs[1]->get_data() == 0) {
        throw std::runtime_error("Division by zero");
    }
    float result = inputs[0]->get_data() / inputs[1]->get_data();
    return std::make_shared<Value>(result, std::vector<std::shared_ptr<Value>>(inputs.begin(), inputs.end()), shared_from_this());
}

void Divide::backward(std::span<const std::shared_ptr<Value>> inputs, std::shared_ptr<const Value> out) const {
    if (inputs.size() != 2) {
        throw std::runtime_error("Divide operation requires exactly two inputs");
    }
    auto out_grad = out->get_grad();
    // y = a/b = a * (1/b)

    // dy/da = 1/b
    inputs[0]->add_grad((1.0f / inputs[1]->get_data()) * out_grad); 
    // dy/db = -a/(b^2) = (a/b) * (1/b)
    inputs[1]->add_grad( (out->get_data()) * (-1.0f / inputs[1]->get_data()) * out_grad);
}


std::string Divide::get_name() const {
    return "/";
}



std::shared_ptr<Value> Exp::forward(std::span<const std::shared_ptr<Value>> inputs) const {
    if (inputs.size() != 1) {
        throw std::runtime_error("Exp operation requires exactly one input");
    }
    float result = std::exp(inputs[0]->get_data());
    return std::make_shared<Value>(result, std::vector<std::shared_ptr<Value>>(inputs.begin(), inputs.end()), shared_from_this());
}

void Exp::backward(std::span<const std::shared_ptr<Value>> inputs, std::shared_ptr<const Value> out) const {
    if (inputs.size() != 1) {
        throw std::runtime_error("Exp operation requires exactly one input");
    }
    auto out_grad = out->get_grad();
    // d(exp(x))/dx = exp(x)
    float exp_x =  out->get_data(); // since out = exp(x)
    inputs[0]->add_grad(exp_x * out_grad);
}

std::string Exp::get_name() const {
    return "exp";
}



float tanh_manual(float x) {
    return (std::exp(2.0f * x) - 1.0f) / (std::exp(2.0f * x) + 1.0f);
}

std::shared_ptr<Value> Tanh::forward(std::span<const std::shared_ptr<Value>> inputs) const {
    if (inputs.size() != 1) {
        throw std::runtime_error("Tanh operation requires exactly one input");
    }
    float result = tanh_manual(inputs[0]->get_data());
    return std::make_shared<Value>(result, std::vector<std::shared_ptr<Value>>(inputs.begin(), inputs.end()), shared_from_this());
}


void Tanh::backward(std::span<const std::shared_ptr<Value>> inputs, std::shared_ptr<const Value> out) const {
    if (inputs.size() != 1) {
        throw std::runtime_error("Tanh operation requires exactly one input");
    }
    // d(tanh(x))/dx = 1 - tanh^2(x)
    auto out_grad = out->get_grad();
    float t = out->get_data(); // tanh(x)
    inputs[0]->add_grad((1.0f - t * t) * out_grad);
}

std::string Tanh::get_name() const {
    return "tanh";
}


namespace operation {

/**
 * public facing APIs for operations
 */
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    static auto add = std::make_shared<Add>(); // to prevent allocating a new Add op each time
    std::array<std::shared_ptr<Value>, 2> inputs{a, b};
    return add->forward(inputs);
}
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    static auto subtract = std::make_shared<Subtract>();
    std::array<std::shared_ptr<Value>, 2> inputs{a, b};
    return subtract->forward(inputs);
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    static auto multiply = std::make_shared<Multiply>();
    std::array<std::shared_ptr<Value>, 2> inputs{a, b};
    return multiply->forward(inputs);
}
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a, const std::shared_ptr<Value> &b)
{
    static auto divide = std::make_shared<Divide>();
    std::array<std::shared_ptr<Value>, 2> inputs{a, b};
    return divide->forward(inputs);
}

std::shared_ptr<Value> tanh(const std::shared_ptr<Value> &x)
{
    static auto tanh_op =  std::make_shared<Tanh>();
    std::array<std::shared_ptr<Value>, 1> inputs{x};
    return tanh_op->forward(inputs);
}


// addition operations for floats directly
std::shared_ptr<Value> operator+(float a, const std::shared_ptr<Value> &b)
{
    return make_value(a) + b;
}
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a, float b)
{
    return a + make_value(b);
}
std::shared_ptr<Value> operator-(float a, const std::shared_ptr<Value> &b)
{
    return make_value(a) - b;
}
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a, float b)
{
    return a - make_value(b);
}
std::shared_ptr<Value> operator*(float a, const std::shared_ptr<Value> &b)
{
    return make_value(a) * b;
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a, float b)
{
    return a * make_value(b);
}
std::shared_ptr<Value> operator/(float a, const std::shared_ptr<Value> &b)
{
    return make_value(a) / b;
}
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a, float b)
{
    return a / make_value(b);
}

std::shared_ptr<Value> exp(const std::shared_ptr<Value> &x){
    static auto exp_op =  std::make_shared<Exp>();
    std::array<std::shared_ptr<Value>, 1> inputs{x};
    return exp_op->forward(inputs);
}

std::shared_ptr<Value> exp(float x){
    return exp(make_value(x));
}

}