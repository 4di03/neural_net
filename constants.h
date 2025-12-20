
#define DEBUG false


// allows us to define debug code that has no effect on runtime when DEBUG is false
#define DBG(x) do { if (DEBUG) { x; } } while (0)
