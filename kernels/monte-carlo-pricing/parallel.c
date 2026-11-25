/*
 * Monte Carlo pricing of a European call - OpenMP version.
 *
 * The M samples are completely independent, so the loop is split into one contiguous share
 * per thread. Each thread accumulates a private sum and sum of squares; at the end the
 * partial results are added up on the main thread.
 *
 * The interesting part is the random number handling. Three modes are provided so the
 * wrong ways can be measured next to the right one:
 *   good      thread w uses its own xoshiro256** stream (base seed advanced by w jumps of
 *             2^128), so streams never overlap and no thread waits for another
 *   sameseed  WRONG: every thread starts from the same state and produces the same numbers,
 *             so the "M samples" are really M/p samples repeated p times; the reported
 *             standard error is then too small by about sqrt(p)
 *   shared    WRONG: one generator for everybody, protected by `omp critical`; correct
 *             statistics but every sample has to queue for the lock
 *
 * Reproducibility: thread w always gets the same share and the same stream, and the partial
 * sums are added in thread order (not in whatever order a reduction clause would pick), so
 * the same seed and thread count always give the same bits. A different thread count uses
 * different streams, so the price changes slightly (by about one standard error).
 *
 * Usage: OMP_NUM_THREADS=p ./parallel M [seed] [good|sameseed|shared]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mc_common.h"

/* per-thread partial results, padded to 64 bytes so two threads never write to the same
 * cache line (otherwise the final stores would cause false sharing) */
typedef struct { double s, s2; char pad[48]; } partial_t;

int main(int argc, char **argv) {
    uint64_t M = argc > 1 ? strtoull(argv[1], 0, 10) : 100000000ULL;
    uint64_t seed = argc > 2 ? strtoull(argv[2], 0, 10) : 2026;
    const char *mode = argc > 3 ? argv[3] : "good";
    int P = omp_get_max_threads();
    partial_t *part = aligned_alloc(64, P * sizeof(partial_t));
    memset(part, 0, P * sizeof(partial_t));
    rng_t shared_rng; rng_stream(&shared_rng, seed, 0);

    double t0 = now_sec();
    #pragma omp parallel
    {
        int w = omp_get_thread_num();
        uint64_t n = M / P + ((uint64_t)w < M % P);   /* my share; the first M%P threads take one extra */
        double s = 0.0, s2 = 0.0;
        if (!strcmp(mode, "shared")) {
            const double drift = (RATE - 0.5 * SIGMA * SIGMA) * TMAT, vol = SIGMA * sqrt(TMAT), disc = exp(-RATE * TMAT);
            for (uint64_t i = 0; i < n; i++) {
                uint64_t a, b;
                #pragma omp critical(rng)
                { a = rng_next(&shared_rng); b = rng_next(&shared_rng); }
                double z = sqrt(-2.0 * log(u01(a))) * cos(6.283185307179586 * u01(b));
                double p = disc * fmax(S0 * exp(drift + vol * z) - KSTR, 0.0);
                s += p; s2 += p * p;
            }
        } else {
            rng_t r;
            rng_stream(&r, seed, !strcmp(mode, "sameseed") ? 0 : w);
            mc_kernel(&r, n, &s, &s2);
        }
        part[w].s = s; part[w].s2 = s2;
    }
    /* combine in a fixed order: floating-point addition is not associative, so a fixed
     * order is what keeps repeated runs bit-identical */
    double sum = 0.0, sum2 = 0.0;
    for (int w = 0; w < P; w++) { sum += part[w].s; sum2 += part[w].s2; }
    double t = now_sec() - t0;

    double mean = sum / M, var = (sum2 / M - mean * mean) * M / (M - 1.0);
    double se = sqrt(var / M), bs = black_scholes_call();
    printf("parallel M=%llu seed=%llu mode=%s threads=%d time=%.6f price=%.10f se=%.3e bs=%.10f err=%.3e err_in_se=%.2f Msamples_per_s=%.2f\n",
           (unsigned long long)M, (unsigned long long)seed, mode, P, t, mean, se, bs, mean - bs, (mean - bs) / se, M / t / 1e6);
    free(part);
    return 0;
}
