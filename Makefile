main: main.cpp autograd.cpp vis.cpp operation.cpp network.cpp
	clang++ -std=c++20 -O2  -g  -Wall -Wextra $^ -o $@
