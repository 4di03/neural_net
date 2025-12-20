main: main.cpp autograd.cpp vis.cpp operation.cpp network.cpp constants.h
	clang++ -std=c++20 -O2 -g -Wall -Wextra \
	main.cpp autograd.cpp vis.cpp operation.cpp network.cpp \
	-o main