/*
 * Shared pieces of the k-mer counter (used by both serial.c and parallel.c).
 *
 *  - make_sequence(): builds a synthetic DNA string from a fixed seed, so every run and
 *    every thread count sees exactly the same input. 'u' gives uniform base frequencies,
 *    's' gives a skewed mix (A 55%, C 20%, G 15%, T 10%) which produces a very uneven
 *    histogram and therefore more collisions between threads.
 *  - ENC[]: 2-bit code per base (A=0, C=1, G=2, T=3). A k-mer is then just a 2k-bit
 *    integer, which doubles as the index into the histogram of 4^k bins.
 *  - COUNT_CHUNK: the actual counting loop, written as a macro so the same code can be
 *    reused with different increment strategies (plain, atomic, private array, ...)
 *    without paying for a function-pointer call per k-mer.
 */
#ifndef KMER_COMMON_H
#define KMER_COMMON_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* 'u' = uniform bases, 's' = skewed (A 55%, C 20%, G 15%, T 10%).
 * One 64-bit random word is split into 32 two-bit draws (uniform) or 8 byte draws (skewed),
 * which keeps generation fast enough for 10^8 bases. */
static char *make_sequence(size_t L, char kind, uint64_t seed) {
    static const char B[4] = {'A', 'C', 'G', 'T'};
    char *s = malloc(L + 1);
    uint64_t st = seed;
    if (kind == 'u') {
        for (size_t i = 0; i < L; i += 32) {
            uint64_t r = splitmix64(&st);
            for (int b = 0; b < 32 && i + b < L; b++) { s[i + b] = B[r & 3]; r >>= 2; }
        }
    } else {
        for (size_t i = 0; i < L; i += 8) {
            uint64_t r = splitmix64(&st);
            for (int b = 0; b < 8 && i + b < L; b++) {
                unsigned x = r & 255; r >>= 8;               /* 0..255 */
                s[i + b] = x < 141 ? 'A' : x < 192 ? 'C' : x < 230 ? 'G' : 'T';
            }
        }
    }
    s[L] = 0;
    return s;
}

/* 2-bit encoding table: A=0 C=1 G=2 T=3 */
static uint8_t ENC[256];
static void init_enc(void) {
    memset(ENC, 0, sizeof ENC);
    ENC['A'] = 0; ENC['C'] = 1; ENC['G'] = 2; ENC['T'] = 3;
}

/* Count all k-mers whose START position lies in [start, end).
 *
 * A rolling window is used: the code of the previous k-mer is shifted left by two bits, the
 * new base is OR-ed in, and the mask drops the oldest base. That is O(1) per k-mer.
 *
 * Chunk boundaries: the loop first "warms up" the window with the k-1 bases that precede
 * the first k-mer of the chunk, and it reads up to position end+k-2. So two neighbouring
 * chunks share k-1 bases, a k-mer that straddles the border is seen only by the chunk that
 * contains its first base, and nothing is counted twice or dropped.
 * INC(idx) is whatever "increment bin idx" means for the caller. */
#define COUNT_CHUNK(seq, start, end, k, mask, INC)                         \
    do {                                                                   \
        uint32_t idx_ = 0;                                                 \
        for (size_t i_ = (start); i_ < (start) + (k) - 1; i_++)            \
            idx_ = (idx_ << 2) | ENC[(uint8_t)(seq)[i_]];                  \
        for (size_t p_ = (start); p_ < (end); p_++) {                      \
            idx_ = ((idx_ << 2) | ENC[(uint8_t)(seq)[p_ + (k) - 1]]) & (mask); \
            INC(idx_);                                                     \
        }                                                                  \
    } while (0)

#endif
