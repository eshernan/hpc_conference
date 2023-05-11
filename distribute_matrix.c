#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#define N 5000

/**
 * Perform matrix multiplication using a mix of shared memory (OpenMP) and distributed memory (MPI).
 */
int main(int argc, char *argv[]) {
    int rank, size;
    int i, j, k;
    double start_time, end_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_local = N / size;
    int *A = (int*) malloc(n_local * N * sizeof(int));
    int *B = (int*) malloc(n_local * N * sizeof(int));
    int *C = (int*) malloc(n_local * N * sizeof(int));
    int *buf = (int*) malloc(N * sizeof(int));

    // Initialize matrices A and B
    for (i = 0; i < n_local; i++) {
        for (j = 0; j < N; j++) {
            A[i*N + j] = i + j + rank * n_local;
            B[i*N + j] = i - j - rank * n_local;
        }
    }

    // Perform matrix multiplication with OpenMP
    start_time = omp_get_wtime();
    #pragma omp parallel for private(j, k)
    for (i = 0; i < n_local; i++) {
        for (j = 0; j < N; j++) {
            C[i*N + j] = 0;
            for (k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
    end_time = omp_get_wtime();

    printf("Execution time on node %d: %f seconds\n", rank, end_time - start_time);

    // Communicate results between nodes using MPI
    if (rank == 0) {
        start_time = MPI_Wtime();
        for (i = 1; i < size; i++) {
            MPI_Recv(buf, N, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (j = 0; j < n_local; j++) {
                for (k = 0; k < N; k++) {
                    C[(i*n_local+j)*N + k] = buf[k];
                }
            }
        }
        end_time = MPI_Wtime();
        printf("Communication time: %f seconds\n", end_time - start_time);
    } else {
        MPI_Send(C, n_local*N, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    free(buf);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
