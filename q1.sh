mpic++ q1.cpp -o q1.out
mpirun --oversubscribe -np "$1" q1.out <"$2"
