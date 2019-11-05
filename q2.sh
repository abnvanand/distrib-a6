mpic++ q2.cpp -o q2.out
mpirun --oversubscribe -np "$1" q2.out <"$2"