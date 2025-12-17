#include "autograd.h"
#include "vis.h"


int main()
{
    auto c = make_value(3, "A") * make_value(-2, "B");
    c -> set_label("C");
    auto e = c + make_value(10, "D");
    e -> set_label("E");
    auto f = make_value(-2.2,"F");
    auto g = e * f;
    g -> set_label("G");


    write_png(g, "graph.png");   // produces graph.png
    return 0;
}