/*
 * Graph input shared by the serial and parallel PageRank programs.
 *
 * Instead of downloading a dataset, the graph is generated with R-MAT (Chakrabarti, Zhan,
 * Faloutsos, "R-MAT: A Recursive Model for Graph Mining", SDM 2004). Each edge is placed by
 * descending `scale` times into one of four quadrants of the adjacency matrix with
 * probabilities (a,b,c,d) = (0.57, 0.19, 0.19, 0.05). The skew gives a power-law degree
 * distribution like real social/web graphs: a few very popular vertices and many with
 * almost no edges.
 *
 * Vertex ids are deliberately NOT shuffled afterwards. In raw R-MAT the busy vertices end up
 * with small ids, so cutting the id range into equal blocks gives some threads far more edges
 * than others - exactly the load-imbalance effect the scheduling experiments look at.
 *
 * The graph is stored twice in CSR (compressed sparse row) form:
 *   out-CSR  edges grouped by source, used by the "push" algorithm
 *   in-CSR   edges grouped by destination, used by the "pull" algorithm
 * CSR = an offset array plus one flat adjacency array, so all neighbours of a vertex are
 * contiguous in memory.
 */
#ifndef GRAPH_H
#define GRAPH_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef struct {
    int n; long m;
    long *out_off; int *out_dst;   /* out-CSR: edges u -> v grouped by u (push) */
    long *in_off;  int *in_src;    /* in-CSR : edges u -> v grouped by v (pull) */
    int *outdeg;
} graph_t;

static inline uint64_t xs64(uint64_t *s) {               /* xorshift64* */
    uint64_t x = *s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Build a CSR structure with a counting sort: count edges per key, prefix-sum the counts into
 * offsets, then drop every edge's value into its slot. key[e] is the vertex the edge is
 * grouped under (source for out-CSR, destination for in-CSR), val[e] is the other endpoint. */
static void build_csr(int n, long m, const int *key, const int *val, long **off_out, int **adj_out) {
    long *off = calloc((size_t)n + 1, sizeof(long));
    int *adj = malloc((size_t)m * sizeof(int));
    for (long e = 0; e < m; e++) off[key[e] + 1]++;
    for (int v = 0; v < n; v++) off[v + 1] += off[v];
    long *pos = malloc((size_t)n * sizeof(long));
    memcpy(pos, off, (size_t)n * sizeof(long));
    for (long e = 0; e < m; e++) adj[pos[key[e]]++] = val[e];
    free(pos);
    *off_out = off; *adj_out = adj;
}

/* R-MAT graph with n = 2^scale vertices and about edgefactor * n edges. Self-loops are
 * dropped (so the final edge count is slightly below the target); duplicate edges are kept,
 * as in the usual R-MAT benchmarks. The thresholds A, B, C below are the cumulative
 * probabilities 0.57, 0.57+0.19, 0.57+0.19+0.19 scaled to 32 bits. */
static graph_t make_rmat(int scale, int edgefactor, uint64_t seed) {
    graph_t g;
    g.n = 1 << scale;
    long target = (long)edgefactor * g.n;
    int *src = malloc(target * sizeof(int)), *dst = malloc(target * sizeof(int));
    uint64_t s = seed ? seed : 1;
    long m = 0;
    const uint32_t A = (uint32_t)(0.57 * 4294967296.0), B = (uint32_t)(0.76 * 4294967296.0),
                   C = (uint32_t)(0.95 * 4294967296.0);
    for (long e = 0; e < target; e++) {
        int u = 0, v = 0;
        for (int l = 0; l < scale; l++) {
            uint32_t r = (uint32_t)(xs64(&s) >> 32);
            int bu = 0, bv = 0;
            if (r < A) { } else if (r < B) bv = 1; else if (r < C) bu = 1; else { bu = 1; bv = 1; }
            u = (u << 1) | bu; v = (v << 1) | bv;
        }
        if (u != v) { src[m] = u; dst[m] = v; m++; }
    }
    g.m = m;
    build_csr(g.n, m, src, dst, &g.out_off, &g.out_dst);
    build_csr(g.n, m, dst, src, &g.in_off, &g.in_src);
    g.outdeg = malloc(g.n * sizeof(int));
    for (int u = 0; u < g.n; u++) g.outdeg[u] = (int)(g.out_off[u + 1] - g.out_off[u]);
    free(src); free(dst);
    return g;
}

#define DAMP 0.85     /* damping factor d of the PageRank formula */
#define EPS  1e-9     /* stop when the L1 change between two iterations drops below this */
#define MAXIT 200     /* safety cap; the test graph converges in about 24 iterations */

#endif
