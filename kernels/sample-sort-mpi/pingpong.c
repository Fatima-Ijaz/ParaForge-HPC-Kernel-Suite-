/*
 * Ping-pong micro-benchmark for the communication cost model T(m) = alpha + beta * m.
 *
 * Rank 0 sends a message of m bytes to rank 1, rank 1 sends it straight back; half of the
 * round-trip time is the one-way time. Message sizes grow by a factor of 4 from 8 bytes to
 * 64 MiB (fewer repetitions for big messages). Small messages are dominated by the latency
 * alpha, large ones by the per-byte cost beta (1/bandwidth); the fit is done later in
 * results/analyze.py and is used to predict the MPI_Alltoallv time of the sample sort.
 *
 * Usage: mpirun -np 2 ./pingpong      prints "bytes one_way_seconds" per message size
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t maxb = (size_t)64 << 20;
    char *buf = calloc(maxb, 1);
    for (size_t m = 8; m <= maxb; m *= 4) {
        int reps = m < 65536 ? 2000 : m < (4u << 20) ? 200 : 20;
        /* the first pass warms up connections and caches, only the second one is printed */
        for (int w = 0; w < 2; w++) {            /* w = 0 warm-up, w = 1 timed */
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int r = 0; r < reps; r++) {
                if (rank == 0) { MPI_Send(buf, (int)m, MPI_CHAR, 1, 0, MPI_COMM_WORLD); MPI_Recv(buf, (int)m, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
                else if (rank == 1) { MPI_Recv(buf, (int)m, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); MPI_Send(buf, (int)m, MPI_CHAR, 0, 0, MPI_COMM_WORLD); }
            }
            double t = (MPI_Wtime() - t0) / (2.0 * reps);
            if (w && rank == 0) printf("%zu %.9e\n", m, t);
        }
    }
    free(buf);
    MPI_Finalize();
    return 0;
}
