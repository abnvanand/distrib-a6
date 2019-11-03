mpic++ q2.cpp -o q2.out
mpirun --oversubscribe -np 4 q2.out < "$1"

