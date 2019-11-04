#include <bits/stdc++.h>
#include<mpi.h>

#define ROOT_ID 0
#define MAX_ITERS 10000

using namespace std;


double dotProduct(const double *x, const double *y, int n);

void computeResidueVector(int myid, double *blockResidueVector, double *blocMatrixA, const double *b, double *x, int n,
                          int n_per_proc);


int main(int argc, char *argv[]) {
    int N, Nplus1;

    int nprocs, myid;
    int num_iters = 0, idx;

    int N_per_proc;
    double *A, *blockA;
    double *b, *x, *blockX;
    double *Ap;
    double rsold, rsnew, block_rsold, bloc_rsnew, *blockR;
    double *p, *blockP, *blockDirectionVector;
    double alpha, finalSum, sum, beta, startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (myid == ROOT_ID) {
        cin >> N >> Nplus1;

        A = (double *) malloc(N * N * sizeof(double *));
        assert(A);
        b = (double *) malloc(N * sizeof(double));
        assert(b);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                cin >> *(A + i * N + j);
            cin >> b[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    MPI_Bcast(&N, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);

    if (N % nprocs != 0) {
        MPI_Finalize();
        if (myid == ROOT_ID)
            cout << "Error : Matrix cannot be evenly distributed N and nprocs are incompatible";;
        exit(-1);
    }

    N_per_proc = N / nprocs;

    auto n_bytes = N * sizeof(double);
    auto n_bytes_per_block = N_per_proc * sizeof(double);

    if (myid != ROOT_ID) {
        b = (double *) malloc(N * sizeof(double));
        assert(b);
    }

    MPI_Bcast(b, N, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

    blockA = (double *) malloc(N_per_proc * n_bytes);
    assert(blockA);

    MPI_Scatter(A, N_per_proc * N, MPI_DOUBLE,
                blockA, N_per_proc * N, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);


    x = (double *) malloc(n_bytes);
    assert(x);
    memset(x, 0, n_bytes);

    blockR = (double *) malloc(n_bytes_per_block);
    assert(blockR);
    computeResidueVector(myid, blockR, blockA, b, x, N,
                         N_per_proc);

    blockP = (double *) malloc(n_bytes_per_block);
    assert(blockP);
    memcpy(blockP, blockR, n_bytes_per_block);   // p=r

    block_rsold = dotProduct(blockR, blockP, N_per_proc);

    blockDirectionVector = (double *) malloc(n_bytes_per_block);
    assert(blockDirectionVector);

    MPI_Allreduce(&block_rsold, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rsold < 1.0e-10) {
        MPI_Finalize();
        exit(0);
    }

    p = (double *) malloc(n_bytes);
    assert(p);

    Ap = (double *) malloc(n_bytes_per_block);
    assert(Ap);

    blockX = (double *) malloc(n_bytes_per_block);
    assert(blockX);

    for (int i = 0; i < N_per_proc; i++)
        blockDirectionVector[i] = -blockP[i];

    for (num_iters = 0; num_iters < MAX_ITERS; num_iters++) {
        MPI_Allgather(blockDirectionVector, N_per_proc, MPI_DOUBLE,
                      p, N_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);

        for (int k = 0; k < N_per_proc; k++) {
            int blockOffset = k * N;
            Ap[k] = dotProduct(p, &blockA[blockOffset], N);
        }

        // p' * Ap
        sum = dotProduct(Ap, blockDirectionVector, N_per_proc);

        sum *= 1.0;
        // MPI_Allreduce is similar to reduce
        // except that the reduced result is available to all procs
        MPI_Allreduce(&sum, &finalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        alpha = rsold;        // alpha = rsold / (p' * Ap)
        alpha /= finalSum;

        for (int index = 0; index < N_per_proc; index++) {
            // r = r - alpha * Ap;
            blockR[index] = blockR[index] + alpha * Ap[index];

            // x = x + alpha * p
            blockX[index] = x[myid * N_per_proc + index] + alpha * blockDirectionVector[index];
        }

        memcpy(blockP, blockR, n_bytes_per_block);   // p=r


        bloc_rsnew = dotProduct(blockR, blockP, N_per_proc);

        MPI_Allgather(blockX, N_per_proc, MPI_DOUBLE, x, N_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);

        // rsnew = r' * r;
        MPI_Allreduce(&bloc_rsnew, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        if (rsnew < 1.0e-10)
            break;


        for (int index = 0; index < N_per_proc; index++) {
            // p = r + (rsnew/rsold) * p
            blockDirectionVector[index] = -blockP[index] + (rsnew / rsold) * blockDirectionVector[index];
        }

        rsold = rsnew;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();

    if (myid == ROOT_ID) {
        cout << "Matrix A " << endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                cout << *(A + i * N + j) << " ";
            cout << endl;
        }

        cout << "Vector B" << endl;
        for (int i = 0; i < N; i++) {
            cout << b[i] << " ";
        }
        cout << endl;

        cout << "Number of steps taken for convergence: " << num_iters << endl;

        cout << "Answer :-" << endl;
        for (int i = 0; i < N; i++) {
            cout << "x" << i << "=" << x[i] << " ";
        }
        cout << endl;

        cout << "CG method took " << endTime - startTime << " seconds." << endl;
    }

    MPI_Finalize();
}

void computeResidueVector(int myid, double *blockResidueVector, double *blocMatrixA, const double *b, double *x, int n,
                          int n_per_proc) {
    int vectorIndex;

    vectorIndex = myid * n_per_proc;
    for (int i = 0; i < n_per_proc; i++) {
        double Ax = dotProduct(&blocMatrixA[i * n], x, n);
        blockResidueVector[i] = Ax - b[vectorIndex++];
    }
}

double dotProduct(const double *x, const double *y, int n) {
    double sum = 0.0;
    double finalSum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x[i] * y[i];

    // MPI_Allreduce is similar to reduce
    // except that the reduced result is available to all procs
//    MPI_Allreduce(&sum, &finalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return sum;
}


