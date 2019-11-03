mpic++ q1.cpp -o q1.out
mpirun --oversubscribe -np 4 q1.out < "$1"

