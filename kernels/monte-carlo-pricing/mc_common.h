/*
 * Shared code of the option pricer: contract parameters, the random number generator, the
 * sampling kernel and the closed-form reference price.
 *
 * The option is a European call on a stock that follows geometric Brownian motion, so the
 * price at maturity can be drawn directly:
 *     S_T = S0 * exp((r - sigma^2/2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)
 * and the price estimate is exp(-rT) * mean(max(S_T - K, 0)) over M independent draws.
 *
 * Random numbers: xoshiro256** (Blackman & Vigna, "Scrambled linear pseudorandom number
 * generators", ACM TOMS 47(4), 2021). Its jump() function advances the state by 2^128 steps,
 * which lets each worker take its own far-apart slice of one long sequence. That gives
 * independent, non-overlapping streams without any locking or shared state.
 * Normal variates come from the Box-Muller transform (two uniforms -> two normals).
 */
#ifndef MC_COMMON_H
#define MC_COMMON_H
#include <stdint.h>
#include <math.h>
#include <time.h>

/* contract / market parameters (the same ones are listed in the report) */
#define S0    100.0
#define KSTR  100.0
#define RATE  0.05
#define SIGMA 0.20
#define TMAT  1.0

static inline double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    uint64_t res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}

/* Advance the generator by 2^128 steps (constants are the published jump polynomial).
 * Calling it w times from a common seed gives stream number w. */
static void rng_jump(rng_t *r) {
    static const uint64_t J[] = {0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
                                 0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL};
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 64; b++) {
            if (J[i] & (1ULL << b)) { s0 ^= r->s[0]; s1 ^= r->s[1]; s2 ^= r->s[2]; s3 ^= r->s[3]; }
            rng_next(r);
        }
    r->s[0] = s0; r->s[1] = s1; r->s[2] = s2; r->s[3] = s3;
}

/* fill the 256-bit state from one 64-bit seed with splitmix64, the usual way to seed xoshiro */
static void rng_seed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        uint64_t z = (seed += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}

/* stream w of a run with a given base seed: seed the generator, then jump w times.
 * The same (seed, w) always gives the same stream, which is what makes runs reproducible. */
static void rng_stream(rng_t *r, uint64_t seed, int w) {
    rng_seed(r, seed);
    for (int i = 0; i < w; i++) rng_jump(r);
}

/* 53 random bits -> double strictly inside (0,1); the +0.5 keeps log(0) out of Box-Muller */
static inline double u01(uint64_t x) { return ((x >> 11) + 0.5) * 0x1.0p-53; }

/* Draw n payoffs from generator r and add them to *sum; add their squares to *sum2.
 * The sum of squares is needed afterwards for the sample variance and the standard error.
 * Samples are generated in pairs because Box-Muller yields two normals per two uniforms. */
static void mc_kernel(rng_t *r, uint64_t n, double *sum, double *sum2) {
    const double drift = (RATE - 0.5 * SIGMA * SIGMA) * TMAT, vol = SIGMA * sqrt(TMAT);
    const double disc = exp(-RATE * TMAT);
    double s = 0.0, s2 = 0.0;
    uint64_t i = 0;
    for (; i + 1 < n; i += 2) {                         /* Box-Muller: 2 normals per pair */
        double u1 = u01(rng_next(r)), u2 = u01(rng_next(r));
        double rad = sqrt(-2.0 * log(u1)), th = 6.283185307179586 * u2;
        double z1 = rad * cos(th), z2 = rad * sin(th);
        double p1 = disc * fmax(S0 * exp(drift + vol * z1) - KSTR, 0.0);
        double p2 = disc * fmax(S0 * exp(drift + vol * z2) - KSTR, 0.0);
        s += p1 + p2; s2 += p1 * p1 + p2 * p2;
    }
    if (i < n) {
        double u1 = u01(rng_next(r)), u2 = u01(rng_next(r));
        double z = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
        double p = disc * fmax(S0 * exp(drift + vol * z) - KSTR, 0.0);
        s += p; s2 += p * p;
    }
    *sum += s; *sum2 += s2;
}

/* Black-Scholes closed form for the same call; used as the reference the estimate is
 * compared with (error and error in units of standard errors are printed). */
static double black_scholes_call(void) {
    double d1 = (log(S0 / KSTR) + (RATE + 0.5 * SIGMA * SIGMA) * TMAT) / (SIGMA * sqrt(TMAT));
    double d2 = d1 - SIGMA * sqrt(TMAT);
    double N1 = 0.5 * erfc(-d1 / sqrt(2.0)), N2 = 0.5 * erfc(-d2 / sqrt(2.0));
    return S0 * N1 - KSTR * exp(-RATE * TMAT) * N2;
}

#endif
