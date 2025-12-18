/**
 * Custom implementation of automatic differentiation on scalar-valued functions, just for fun and learning.
 */
#include <iostream>
#include "autograd.h"
#include "operation.h"

#define DEBUG false


// allows us to define debug code that has no effect on runtime when DEBUG is false
#define DBG(x) do { if (DEBUG) { x; } } while (0)


std::ostream &operator<<(std::ostream &os, const Operation &op){
   os << op.get_name();
    return os;
}


// cout overload for shared_ptr<Value>
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Value> &v)
{
    os << "\nValue(data=" << v->get_data() << ", operation = " << (v->get_operation() != nullptr ? v->get_operation()->get_name() : "nullopt") << ", prev=[";
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






std::vector<std::shared_ptr<Value>> topo_sort(const std::shared_ptr<Value>& out){
    // top-sort with cycle detection, where the first node has no ancestors while the last node has the most ancestors
    std::unordered_map<std::shared_ptr<Value>, int> in_degree;

    // count the in-degrees of each node using dfs
    std::function<void(const std::shared_ptr<Value>&)> dfs_count =
        [&](const std::shared_ptr<Value>& v) {
            if (!v) return;

            if (in_degree.count(v)) return; // already visited

            in_degree[v] = 0; // initialize in-degree

            for (const auto& p : v->get_prev()) {
                dfs_count(p);
                in_degree[p]++; // count incoming edge
            }
        };

    dfs_count(out);

    std::vector<std::shared_ptr<Value>> sorted;
    sorted.reserve(in_degree.size());
    size_t cur_index = 0;

    for (const auto& [ptr, val] : in_degree) {
        if (val == 0) { // get in-degree 0 nodes
           sorted.push_back(ptr);
        }
    }

    while (cur_index < sorted.size()) { // keep going while we have nodes left to process in the topological order
        auto v = sorted[cur_index++];

        // relax the child edges
        for (const auto& p : v->get_prev()) {
            in_degree[p]--;
            if (in_degree[p] == 0) {
                sorted.push_back(p);
            }
        }
    }

    if (sorted.size() != in_degree.size()) {
        throw std::runtime_error("Cycle detected in computation graph during topological sort, sorted.size: " + std::to_string(sorted.size()) + ", in_degree.size(): " + std::to_string(in_degree.size()) + ")");
    }

    
    DBG(
    // print sorted list by values
    for (const auto& v : sorted) {
            std::cout << "Topological sort node: Value(data=" << v->get_data() << ", operation = " << (v->get_operation() != nullptr ? v->get_operation()->get_name() : "nullopt") << ")\n";
    }
    );

    return sorted;
}

void Value::backward()
{
    // topological sort the computation graph starting from this node
    auto sorted = topo_sort(shared_from_this());

    // set the gradient of this node w.r.t itself to be 1.0
    this->set_grad(1.0f);

    // traverse in topological order to propagate gradients from end to start of comp graph 
    // we do it top-down because if y = f(g(x)) , then dy/dx = dy/dg * dg/dx, so we need to know dy/dg before we can compute dy/dx
    for (auto it = sorted.begin(); it != sorted.end(); ++it) {
        auto v = *it;
        auto op = v->get_operation();
        DBG(
        std::cout << "Backpropagating through Value node with data=" << v->get_data() << ", grad=" << v->get_grad() << ", operation=" << (op != nullptr ? op->get_name() : "nullopt") << "\n";
        );
        if (op != nullptr) {
            // make span of prev
            auto prev_span = std::span<std::shared_ptr<Value> const>(v->get_prev().data(), v->get_prev().size());
            op->backward(prev_span, v);
        }
    }

}