/*
 * Monte Carlo pricing of a European call - MPI version.
 *
 * Same estimator as parallel.c, written for distributed memory so the two programming
 * models can be compared on one problem. Rank w takes the same share of the samples and
 * the same random stream w as OpenMP thread w. Each rank sums its own samples, the two
 * partial sums (value and value squared) are collected on rank 0 with MPI_Gather and added
 * in rank order. Because the partition, the streams and the summation order are identical,
 * the price is bit-for-bit the same as the OpenMP run with the same number of workers.
 *
 * Usage: mpirun -np p ./parallel_mpi M [seed]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "mc_common.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int w, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &w);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    uint64_t M = argc > 1 ? strtoull(argv[1], 0, 10) : 100000000ULL;
    uint64_t seed = argc > 2 ? strtoull(argv[2], 0, 10) : 2026;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    uint64_t n = M / P + ((uint64_t)w < M % P);
    rng_t r; rng_stream(&r, seed, w);
    double loc[2] = {0.0, 0.0};
    mc_kernel(&r, n, &loc[0], &loc[1]);
    /* only 16 bytes per rank travel over the network, so communication cost is negligible */
    double *all = w == 0 ? malloc(2 * P * sizeof(double)) : NULL;
    MPI_Gather(loc, 2, MPI_DOUBLE, all, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime() - t0;

    if (w == 0) {
        double sum = 0.0, sum2 = 0.0;
        for (int q = 0; q < P; q++) { sum += all[2 * q]; sum2 += all[2 * q + 1]; }
        double mean = sum / M, var = (sum2 / M - mean * mean) * M / (M - 1.0);
        double se = sqrt(var / M), bs = black_scholes_call();
        printf("mpi M=%llu seed=%llu ranks=%d time=%.6f price=%.10f se=%.3e bs=%.10f err=%.3e Msamples_per_s=%.2f\n",
               (unsigned long long)M, (unsigned long long)seed, P, t, mean, se, bs, mean - bs, M / t / 1e6);
        free(all);
    }
    MPI_Finalize();
    return 0;
}
