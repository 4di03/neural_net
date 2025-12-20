simple scalar autograd engine in C++


## Setup:
We don't use any libraries beyond the C++ standard library. If you'd like to visualize the computation graph via PNG, you'll need to have Graphviz installed.

- Requires C++20
- Requires Graphviz installed for PNG output (optional)

ideas:
- make more efficient backprop using auto-differentiation via recursive top-down gradient calculation
- extend to tensors
- add python bindings

