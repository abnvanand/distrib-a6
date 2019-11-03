#include <bits/stdc++.h>
#include <mpi.h>

#define MAX_LEN 1000
#define ROOT_ID 0
using namespace std;

/**
 * Input Format :-
 * N, N+1
 * a00 a01..a0n-1 b0
 * a10 a11....... b1
 * .............. ..
 * .............. bn-1
 */

double dotProduct(const double *x, const double *y, int n);

void axpy(double *res, double x, double *A, double *y, int n);

void matMulVec(double *res, double **A, double *v, int n);


/**STEPS of ALGO
 *
 * Step 1. Read the data from the input file and divide it across processors,
 * using MPI Bcast and MPI Scatter(v).
 *
 * Step 2. Compute the inner product locally, and do a sum reduction
 * (MPI Allreduce) across all processes.
 *
 * Step 3. Do the matrix-vector product in parallel using MPI Allgatherv,
 * to gather all the local parts of the vector into a single vector, and then to
 * do the multiplication.
 *
 * Details of each step:-
 *Step 1: the algorithm first loads the data in one of the
 * processors, uses MPI Bcast to send the dimension of the matrix to the other
 * processors, and then ”divides” the data into subsets that are sent to the other
 * processors via MPI Scatterv and MPI Scatter.
 *
 * Step 2: The inner product is calculated on parallel by computing a local sum on each
 * processor and then doing a sum reduction, using MPI Allreduce.
 * see function `dot()`
 */

void conjugrad(double **A, double *b, double *x, int n) {
    const int nbytes = n * sizeof(double);

    double *p;
    double *r;
    double *Ap;
    double *Ax;

    double rsnew, rsold, alpha;

    p = (double *) malloc(nbytes);
    assert (p);
    r = (double *) malloc(nbytes);
    assert (r);
    Ap = (double *) malloc(nbytes);
    assert (Ap);
    Ax = (double *) malloc(nbytes);
    assert (Ax);

    memset(x, 0, nbytes);
    matMulVec(Ax, A, x, n);

    axpy(r, -1, Ax, b, n);  // r = b - Ax
    memcpy(p, r, nbytes);   // p=r

    rsold = dotProduct(r, r, n); // rsold= r' * r


    for (int i = 0; i < 10; i++) {
        cout << "Iter " << i << " ";
        for (int k = 0; k < n; k++)
            cout << x[k] << " ";
        cout << endl;

        matMulVec(Ap, A, p, n);  // Ap = A * p
        alpha = rsold / dotProduct(p, Ap, n); // alpha = rsold / (p' * Ap)
        axpy(x, alpha, p, x, n); // x = x + alpha * p
        axpy(r, -alpha, Ap, r, n); // r = r - alpha * Ap;
        rsnew = dotProduct(r, r, n);   // rsnew = r' * r;

        if (sqrt(rsnew) < 1e-10) {
            break;
        }

        axpy(p, rsnew / rsold, p, r, n);    // p = r + (rsnew/rsold) * p
        rsold = rsnew;
    }
}

int main() {
    MPI_Init(nullptr, nullptr);

    double *b, *x;
    double **A;

    // TODO: Understand usage
    double c[MAX_LEN];
    int map[MAX_LEN];
    double sum = 0.0;
    // STOPSHIP


    int N, Nplus1;

    int my_id, n_procs;
    double startTime = 0, endTime = 0;


    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    ///////////////////// Read input BEGIN /////////////////////
    if (my_id == ROOT_ID) {
        cin >> N >> Nplus1;

        A = (double **) malloc(N * sizeof(double));
        b = (double *) malloc(N * sizeof(double));
        x = (double *) malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            A[i] = (double *) malloc(N * sizeof(double));
            for (int j = 0; j < N; j++) {
                cin >> A[i][j];
            }
            cin >> b[i];
            x[i] = 0;
        }

        cout << "Matrix A" << endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                cout << A[i][j] << " ";
            cout << endl;
        }
        cout << "Vector B" << endl;
        for (int i = 0; i < N; i++)
            cout << b[i] << " ";
        cout << endl;
    }
    ///////////////////// Read input END /////////////////////

    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast array dimension (N)
    MPI_Bcast(&N, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
    cout << "my_id=" << my_id << " N=" << N << endl;


//    MPI_Bcast(&A, N * N, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
//    MPI_Bcast(&b, N, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
//    MPI_Bcast(&x, N, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

    if (my_id == ROOT_ID) {
        startTime = MPI_Wtime();
        conjugrad(A, b, x, N);
        endTime = MPI_Wtime();
    }

//    if (my_id == ROOT_ID) {
//        for (int i = 0; i < N; i++) {
//            for (int j = 0; j < N; j++)
//                cout << A[i][j] << " ";
//            cout << endl;
//        }
//    }


    if (my_id == ROOT_ID) {
        cout << "Answer :-" << endl;
        for (int i = 0; i < N; i++) {
            cout << "x" << i << "=" << x[i] << " ";
        }
        cout << endl;

        cout << "CG Method time: " << endTime - startTime << endl;
        free(A);
        free(b);
        free(x);
    }

    MPI_Finalize();
    return 0;
}

/*
 * Function to calculate the inner product.
 * The inner product is calculated in parallel by computing a local sum on each
 * processor and then doing a sum reduction, using MPI Allreduce.
 */
double dotProduct(const double *x, const double *y, int n) {
    int i;
    double sum = 0.0;
    double finalSum = 0.0;
    for (i = 0; i < n; ++i)
        sum += x[i] * y[i];

    // MPI_Allreduce is similar to reduce
    // except that the reduced result is available to all procs
//    MPI_Allreduce(&sum, &finalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sum;
}

void axpy(double *res, double x, double *A, double *y, int n) {
    // res = Ax + y
    for (int i = 0; i < n; i++)
        res[i] = (A[i] * x) + y[i];
}

void matMulVec(double *res, double **A, double *v, int n) {
    for (int i = 0; i < n; i++) {
        res[i] = 0;
        for (int j = 0; j < n; j++) {
            res[i] += A[i][j] * v[j];
        }
    }
}


//void matMulVec(double *y, double **A, double *xsend, int m) {
//    int *rowInd = A[0];
//    int *colInd = 0;
//    double *localval = A[0][0];
//    double xrecv[DIM], elemA;
//    int col;
//    MPI_Allgatherv(xsend, m, MPI_DOUBLE, xrecv, rowCounts,
//                   rowDispl, MPI_DOUBLE, MPI_COMM_WORLD);
//    for (int i = 0; i < m; ++i) {
//        y[i] = 0;
//        for (int j = rowInd[i]; j < rowInd[i + 1]; ++j) {
//            elemA = *localVal++;
//            col = *colInd++;
//            y[i] = y[i]
//            elemA * xrecv[col];
//        }
//    }
//}