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
int main() {
    MPI_Init(nullptr, nullptr);

    double A[MAX_LEN][MAX_LEN], b[MAX_LEN], x[MAX_LEN];

    double c[MAX_LEN];
    int map[MAX_LEN];
    double sum = 0.0;


    int N, Nplus1;

    int my_id, n_procs;
    double timer_forward_begin, timer_forward_end, timer_backward_begin, timer_backward_end = 0;

//    vector<vector<double>> A(N, vector<double>(N, 0));
//    vector<double> b(N, 0);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    ///////////////////// Read input BEGIN /////////////////////
    if (my_id == ROOT_ID) {
        cin >> N >> Nplus1;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cin >> A[i][j];
            }
            cin >> b[i];
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

    ////////////////////// Gaussian elimination (BEGIN) ////////////////////

    timer_forward_begin = MPI_Wtime();

    // Broadcast matrix A
    // NOTE: don't broadcast count = N*N bcoz each array size is actually MAX_LEN
    // So 1st part will be broadcasted correctly but rest of the procs will get garbage part of the array
    MPI_Bcast(&A[0][0], MAX_LEN * MAX_LEN, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

    // Broadcast vector B
    // Above problem does not apply here becoz B is a 1d array
    MPI_Bcast(b, N, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

    // Map which part to be processed by which process
    for (int i = 0; i < N; i++) {
        map[i] = i % n_procs;
    }

    for (int k = 0; k < N; k++) {
        MPI_Bcast(&A[k][k], N - k, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
        for (int i = k + 1; i < N; i++) {
            if (map[i] == my_id) { // check whether current process should compute this part
                c[i] = A[i][k] / A[k][k];
            }
        }

        for (int i = k + 1; i < N; i++) {
            if (map[i] == my_id) {
                for (int j = 0; j < N; j++) {
                    A[i][j] = A[i][j] - (c[i] * A[k][j]);
                }
                b[i] = b[i] - (c[i] * b[k]);
            }
        }
    }
    timer_forward_end = MPI_Wtime();

    ////////////////////// Gaussian elimination (END) ////////////////////


    ////////////////////// Back substitution (BEGIN) ////////////////////
    timer_backward_begin = MPI_Wtime();

    if (my_id == ROOT_ID) {
        x[N - 1] = b[N - 1] / A[N - 1][N - 1];
        for (int i = N - 2; i >= 0; i--) {
            sum = 0;

            for (int j = i + 1; j < N; j++) {
                sum = sum + A[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i][i];
        }

        timer_backward_end = MPI_Wtime();
    }

    ////////////////////// Back substitution (END) ////////////////////
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_id == ROOT_ID) {
        cout << "Answer :-" << endl;
        for (int i = 0; i < N; i++) {
            cout << "x" << i << "=" << x[i] << " ";
        }
        cout << endl;

        cout << "Gaussian elimination time: " << timer_forward_end - timer_forward_begin << endl;
        cout << "Back substitution time: " << timer_backward_end - timer_backward_begin << endl;
        cout << "Total time: " << timer_backward_end - timer_forward_begin << endl;
    }

    MPI_Finalize();
    return 0;
}
