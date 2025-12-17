#include "autograd.h"
#include "vis.h"


int main()
{
    auto expr = make_value(3) * make_value(-2) + make_value(10);
    write_png(expr, "graph.png");   // produces graph.png
    return 0;
}