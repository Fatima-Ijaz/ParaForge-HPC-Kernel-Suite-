/*
 * k-mer counting - OpenMP version, with several ways of handling the shared histogram.
 *
 * The work (one increment per k-mer position) is trivially splittable, the hard part is
 * that many threads want to bump the same histogram bin. This program implements the
 * different answers to that side by side so they can be timed against each other:
 *
 *   A  plain parallel for, hist[i]++ with no protection. This is WRONG on purpose: two
 *      threads can read the same old value, both add one and both store the same result,
 *      so increments get lost. It exists to show the size of the damage.
 *   B  same loop, but every increment is an `omp atomic`. Correct, but each increment
 *      becomes a locked read-modify-write and the cache lines ping-pong between cores.
 *   C  every thread counts into its own private histogram, then the copies are added up
 *      in parallel (each thread sums a slice of the bins). No sharing while counting, at
 *      the price of P * 4^k * 4 bytes of extra memory.
 *   R  the same idea as C, but delegated to OpenMP's array-section reduction.
 *   F  false-sharing experiment: private counters for all threads are packed in one array
 *      with a configurable stride. With stride 0 the counters of different threads sit in
 *      the same 64-byte cache line (so the line bounces although nobody shares a counter);
 *      with a padded stride each thread gets its own line.
 *
 * Work division: the k-mer start positions are cut into chunks of 65536 and the chunks are
 * handed out with schedule(static). Each chunk also reads k-1 bases past its end (see
 * COUNT_CHUNK in kmer_common.h), so k-mers that cross a chunk border are counted once.
 *
 * Usage: OMP_NUM_THREADS=p ./parallel L k [u|s] VERSION [pad]
 *   VERSION  A | B | C | R | F
 *   pad      only for F: 0 = adjacent counters, 16 = one 64-byte line per thread
 *
 * After timing, the result is compared bin by bin with a serial recount (not timed) and
 * the line ends with PASS or FAIL; `lost` shows how many increments went missing.
 */
#include <stdio.h>
#include <omp.h>
#include "kmer_common.h"

/* k-mer start positions per loop iteration: big enough to amortise loop overhead,
 * small enough (about 100 chunks per thread at L = 10^8) to keep threads balanced */
#define CHUNK ((size_t)1 << 16)

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s L k u|s A|B|C|R|F [pad]\n", argv[0]); return 1; }
    size_t L = strtoull(argv[1], 0, 10);
    int k = atoi(argv[2]);
    char kind = argv[3][0], ver = argv[4][0];
    int pad = argc > 5 ? atoi(argv[5]) : 0;
    int P = omp_get_max_threads();

    init_enc();
    char *seq = make_sequence(L, kind, 12345);
    size_t nb = (size_t)1 << (2 * k);
    uint32_t mask = (uint32_t)(nb - 1);
    size_t nk = L - k + 1;
    long nchunks = (long)((nk + CHUNK - 1) / CHUNK);

    uint32_t *h = calloc(nb, sizeof(uint32_t));
    double extra_mem = 0.0;
    double t0 = now_sec();

    if (ver == 'A') {
#define INC_RACE(i) h[i]++
        #pragma omp parallel for schedule(static)
        for (long c = 0; c < nchunks; c++) {
            size_t s = c * CHUNK, e = s + CHUNK < nk ? s + CHUNK : nk;
            COUNT_CHUNK(seq, s, e, k, mask, INC_RACE);
        }
    } else if (ver == 'B') {
#define INC_ATOMIC(i) _Pragma("omp atomic") h[i]++
        #pragma omp parallel for schedule(static)
        for (long c = 0; c < nchunks; c++) {
            size_t s = c * CHUNK, e = s + CHUNK < nk ? s + CHUNK : nk;
            COUNT_CHUNK(seq, s, e, k, mask, INC_ATOMIC);
        }
    } else if (ver == 'C') {
        /* version C: one private histogram per thread */
        uint32_t **priv = malloc(P * sizeof(uint32_t *));
        extra_mem = (double)P * nb * sizeof(uint32_t);
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            uint32_t *mine = priv[t] = calloc(nb, sizeof(uint32_t));  /* first touch by owner */
#define INC_PRIV(i) mine[i]++
            #pragma omp for schedule(static)
            for (long c = 0; c < nchunks; c++) {
                size_t s = c * CHUNK, e = s + CHUNK < nk ? s + CHUNK : nk;
                COUNT_CHUNK(seq, s, e, k, mask, INC_PRIV);
            }
            /* the barrier at the end of the loop above guarantees all copies are complete;
             * now each thread adds up its own slice of bins across all copies */
            #pragma omp for schedule(static)
            for (size_t b = 0; b < nb; b++) {
                uint32_t acc = 0;
                for (int q = 0; q < P; q++) acc += priv[q][b];
                h[b] = acc;
            }
        }
        for (int q = 0; q < P; q++) free(priv[q]);
        free(priv);
    } else if (ver == 'R') {
        /* version R: the compiler creates the private copies and merges them itself */
        extra_mem = (double)P * nb * sizeof(uint32_t);
#define INC_RED(i) h[i]++
        #pragma omp parallel for schedule(static) reduction(+ : h[:nb])
        for (long c = 0; c < nchunks; c++) {
            size_t s = c * CHUNK, e = s + CHUNK < nk ? s + CHUNK : nk;
            COUNT_CHUNK(seq, s, e, k, mask, INC_RED);
        }
    } else if (ver == 'F') {
        /* version F: thread t uses counters shared[t*stride .. t*stride + nb); with the
         * smallest stride (= nb) neighbouring threads' counters share cache lines */
        size_t stride = pad ? (size_t)pad : nb;
        if (stride < nb) stride = nb;
        volatile uint32_t *shared = aligned_alloc(64, ((P * stride * 4 + 63) / 64) * 64);
        memset((void *)shared, 0, P * stride * 4);
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            volatile uint32_t *mine = shared + (size_t)t * stride;
#define INC_FS(i) mine[i]++
            #pragma omp for schedule(static)
            for (long c = 0; c < nchunks; c++) {
                size_t s = c * CHUNK, e = s + CHUNK < nk ? s + CHUNK : nk;
                COUNT_CHUNK(seq, s, e, k, mask, INC_FS);
            }
            #pragma omp for
            for (size_t b = 0; b < nb; b++) {
                uint32_t acc = 0;
                for (int q = 0; q < P; q++) acc += shared[(size_t)q * stride + b];
                h[b] = acc;
            }
        }
        free((void *)shared);
    } else { fprintf(stderr, "unknown version\n"); return 1; }

    double t = now_sec() - t0;

    /* ---- verification against a serial reference (not timed) ----
     * the total must be L-k+1 and every bin must match; version A is expected to fail here */
    uint32_t *ref = calloc(nb, sizeof(uint32_t));
#define INC_REF(i) ref[i]++
    COUNT_CHUNK(seq, (size_t)0, nk, k, mask, INC_REF);
    uint64_t total = 0; size_t bad = 0;
    for (size_t b = 0; b < nb; b++) { total += h[b]; bad += (h[b] != ref[b]); }

    printf("parallel L=%zu k=%d seq=%c ver=%c pad=%d threads=%d time=%.6f total=%llu expected=%zu lost=%lld bad_bins=%zu extra_MB=%.1f Mkmers_per_s=%.1f %s\n",
           L, k, kind, ver, pad, P, t, (unsigned long long)total, nk, (long long)nk - (long long)total, bad,
           extra_mem / 1048576.0, nk / t / 1e6, bad ? "FAIL" : "PASS");
    free(ref); free(h); free(seq);
    return 0;
}
