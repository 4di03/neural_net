/**
 * Custom implementation of a simple neural network, just for fun and learning.
 * 
 * For autograd, we approximate deirvatives using finite differences.
 */
#include <iostream>

float f(int x) {
    return x * x;
}
template <class T>
void print(const T& x) {
    std::cout << x << '\n';   // works for anything with operator<<
}




class Value {
public:
    float data;
    std::vector<std::shared_ptr<Value>> children; // keep dependent values for autograd
    Value(float data) : data(data) {}
    Value(float data, const std::vector<std::shared_ptr<Value>>& children) : data(data), children(children) {}
};

/**
 * We  set the children to be the operands involved in the operation, so taht we can trace back during backpropagation.
 * 
 * WE only define this on shared pointers to make sure we don't copy around values and miss updating dependencies in the computation graph.
 */
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a->data + b->data, std::vector<std::shared_ptr<Value>>{ a, b } );
}
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a->data - b->data, std::vector<std::shared_ptr<Value>>{ a, b } );
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a->data * b->data, std::vector<std::shared_ptr<Value>>{ a, b } );
}
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    if (b->data == 0) {
        throw std::runtime_error("Division by zero");
    }
    return std::make_shared<Value>(a->data / b->data, std::vector<std::shared_ptr<Value>>{ a, b } );
}

//cout overload for shared_ptr<Value>
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& v) {
    os << v->data;
    os << " (children: ";
    for (const auto& child : v->children) {
        os << child->data << " ";
    }
    os << ")";
    return os;
}


/**
 * Approximate derivative using (f(x + h)) - f(x)) / h for a small x 
 * 
 * 
 * Allows use to compute derivates for general functions of from f(x), where f is any callable object
 */
std::shared_ptr<Value> make_value(float x) {
    return std::make_shared<Value>(x);
}


int main(){
    print(make_value(5) + make_value(10));
    print(make_value(5) * make_value(2));
    print(make_value(5) / make_value(2));
    print(f(5));
    print(make_value(3));
    return 0;
 }