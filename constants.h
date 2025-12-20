
#define DEBUG false
#define DRAW_GRAPHS true // whether to draw computation graphs or not
#define LEARNING_RATE 0.05f
#define N_EPOCHS 100


// allows us to define debug code that has no effect on runtime when DEBUG is false
#define DBG(x) do { if constexpr (DEBUG) { x; } } while (0)
