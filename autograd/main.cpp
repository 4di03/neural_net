#include "autograd.h"
#include "vis.h"


int main()
{


    auto a = make_value(2.0, "A");
    auto b = make_value(-3.0, "B");

    auto c = a * b;
    c -> set_label("C");
    auto d = make_value(10.0, "D");

    auto e = c + d;
    e -> set_label("E");
    auto f = make_value(-2.2,"F");

    auto g = e * f;
    g -> set_label("G");

    write_png(g, "graph.png");   // produces graph.png
    return 0;
}
