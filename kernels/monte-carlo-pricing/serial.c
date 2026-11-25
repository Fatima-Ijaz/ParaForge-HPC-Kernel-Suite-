/*
 * Monte Carlo pricing of a European call - serial reference version.
 *
 * One generator, M samples, one loop. It prints the price estimate, its standard error
 * (sqrt(sample variance / M)), the Black-Scholes price and how many standard errors the
 * two are apart. The parallel versions are timed against this program and their price must
 * agree with it (bit for bit when run with one worker).
 *
 * Usage: ./serial M [seed]     (defaults: M = 10^8, seed = 2026)
 */
#include <stdio.h>
#include <stdlib.h>
#include "mc_common.h"

int main(int argc, char **argv) {
    uint64_t M = argc > 1 ? strtoull(argv[1], 0, 10) : 100000000ULL;
    uint64_t seed = argc > 2 ? strtoull(argv[2], 0, 10) : 2026;
    rng_t r; rng_stream(&r, seed, 0);
    double sum = 0.0, sum2 = 0.0;
    double t0 = now_sec();
    mc_kernel(&r, M, &sum, &sum2);
    double t = now_sec() - t0;
    double mean = sum / M, var = (sum2 / M - mean * mean) * M / (M - 1.0);
    double se = sqrt(var / M), bs = black_scholes_call();
    printf("serial M=%llu seed=%llu time=%.6f price=%.10f se=%.3e bs=%.10f err=%.3e err_in_se=%.2f Msamples_per_s=%.2f\n",
           (unsigned long long)M, (unsigned long long)seed, t, mean, se, bs, mean - bs, (mean - bs) / se, M / t / 1e6);
    return 0;
}
