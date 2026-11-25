/*
 * k-mer counting - serial reference version.
 *
 * Generates a DNA sequence of L bases, slides a window of length k over it and counts how
 * often each k-mer occurs. The result is a histogram with 4^k bins (each base takes 2 bits,
 * so a k-mer is directly an array index). The timing covers the counting loop only;
 * generating the sequence is reported separately as gen_time.
 *
 * Usage: ./serial L k [u|s] [histfile]
 *   L         sequence length in bases (default 10^8)
 *   k         k-mer length (the report uses 8 and 12)
 *   u|s       uniform or skewed base frequencies
 *   histfile  optional; write the histogram (4^k uint32 values) to a file
 *
 * The "hash" in the output is a fingerprint of the whole histogram; the parallel program
 * does a bin-by-bin comparison against its own serial recount instead.
 */
#include <stdio.h>
#include "kmer_common.h"

int main(int argc, char **argv) {
    size_t L = argc > 1 ? strtoull(argv[1], 0, 10) : 100000000ULL;
    int k = argc > 2 ? atoi(argv[2]) : 8;
    char kind = argc > 3 ? argv[3][0] : 'u';
    const char *out = argc > 4 ? argv[4] : NULL;

    init_enc();
    double g0 = now_sec();
    char *seq = make_sequence(L, kind, 12345);
    double gen = now_sec() - g0;

    size_t nb = (size_t)1 << (2 * k);
    uint32_t mask = (uint32_t)(nb - 1);
    uint32_t *h = calloc(nb, sizeof(uint32_t));
    size_t nk = L - k + 1;

    double t0 = now_sec();
#define INC_SERIAL(i) h[i]++
    COUNT_CHUNK(seq, (size_t)0, nk, k, mask, INC_SERIAL);
    double t = now_sec() - t0;

    uint64_t total = 0, chk = 0;
    for (size_t b = 0; b < nb; b++) { total += h[b]; chk = chk * 1000003ULL + h[b]; }
    printf("serial L=%zu k=%d seq=%c time=%.6f gen_time=%.3f total=%llu expected=%zu hash=%016llx Mkmers_per_s=%.1f\n",
           L, k, kind, t, gen, (unsigned long long)total, nk, (unsigned long long)chk, nk / t / 1e6);
    if (out) { FILE *f = fopen(out, "wb"); fwrite(h, sizeof(uint32_t), nb, f); fclose(f); }
    free(h); free(seq);
    return 0;
}
