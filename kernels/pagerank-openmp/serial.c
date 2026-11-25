/*
 * PageRank - serial reference version on a CSR graph.
 *
 * Each iteration computes, for every vertex v,
 *     r_new[v] = (1 - d)/n + d * ( sum over in-neighbours u of r[u]/outdeg(u)  +  dangling/n )
 * with d = 0.85, and stops when the L1 distance between r_new and r drops below 1e-9.
 * "Dangling" vertices have no out-edges, so their rank would simply vanish; their combined
 * rank is spread evenly over all vertices instead, which keeps the ranks summing to 1.
 *
 * This is the pull formulation: v reads from its in-neighbours, so every output element is
 * written by exactly one loop iteration. Per iteration, c[u] = r[u]/outdeg(u) is computed
 * once so the inner loop is just a sum of gathered values.
 *
 * Usage: ./serial scale edgefactor [ranks.bin] [degrees.txt]
 *   scale, edgefactor  R-MAT size: 2^scale vertices, about edgefactor * 2^scale edges
 *   ranks.bin          optional; save the final ranks so ./parallel can verify against them
 *   degrees.txt        optional; in/out degree histogram for the log-log plot
 */
#include <stdio.h>
#include <math.h>
#include "graph.h"

int main(int argc, char **argv) {
    int scale = argc > 1 ? atoi(argv[1]) : 20, ef = argc > 2 ? atoi(argv[2]) : 16;
    const char *rout = argc > 3 ? argv[3] : NULL, *dout = argc > 4 ? argv[4] : NULL;
    double g0 = now_sec();
    graph_t g = make_rmat(scale, ef, 42);
    double gen = now_sec() - g0;
    int n = g.n;

    if (dout) {   /* in- and out-degree histograms (how many vertices have degree d) */
        int maxd = 0;
        for (int v = 0; v < n; v++) { int d = g.in_off[v + 1] - g.in_off[v]; if (d > maxd) maxd = d; if (g.outdeg[v] > maxd) maxd = g.outdeg[v]; }
        long *hi = calloc(maxd + 1, sizeof(long)), *ho = calloc(maxd + 1, sizeof(long));
        for (int v = 0; v < n; v++) { hi[g.in_off[v + 1] - g.in_off[v]]++; ho[g.outdeg[v]]++; }
        FILE *f = fopen(dout, "w");
        fprintf(f, "degree in_count out_count\n");
        for (int d = 0; d <= maxd; d++) if (hi[d] || ho[d]) fprintf(f, "%d %ld %ld\n", d, hi[d], ho[d]);
        fclose(f); free(hi); free(ho);
    }

    double *r = malloc(n * sizeof(double)), *rn = malloc(n * sizeof(double)), *c = malloc(n * sizeof(double));
    for (int v = 0; v < n; v++) r[v] = 1.0 / n;
    int it; double diff = 0.0;
    double t0 = now_sec();
    for (it = 1; it <= MAXIT; it++) {
        /* step 1: per-vertex contribution, and the total rank sitting on dangling vertices */
        double dang = 0.0;
        for (int u = 0; u < n; u++) {
            if (g.outdeg[u]) c[u] = r[u] / g.outdeg[u]; else { c[u] = 0.0; dang += r[u]; }
        }
        /* teleport term + the dangling mass shared equally; identical for every vertex */
        double base = (1.0 - DAMP) / n + DAMP * dang / n;
        diff = 0.0;
        /* step 2: gather from in-neighbours (the random reads of c[] dominate the cost) */
        for (int v = 0; v < n; v++) {
            double s = 0.0;
            for (long e = g.in_off[v]; e < g.in_off[v + 1]; e++) s += c[g.in_src[e]];
            rn[v] = base + DAMP * s;
            diff += fabs(rn[v] - r[v]);
        }
        double *t = r; r = rn; rn = t;
        if (diff < EPS) break;
    }
    double t = now_sec() - t0;
    int dangling = 0; for (int u = 0; u < n; u++) dangling += !g.outdeg[u];
    double sum = 0.0; for (int v = 0; v < n; v++) sum += r[v];
    printf("serial scale=%d ef=%d n=%d m=%ld dangling=%d gen_time=%.2f iters=%d time=%.6f time_per_iter=%.6f l1=%.3e rank_sum=%.12f Medges_per_s=%.1f\n",
           scale, ef, n, g.m, dangling, gen, it, t, t / it, diff, sum, (double)g.m * it / t / 1e6);
    if (rout) { FILE *f = fopen(rout, "wb"); fwrite(r, sizeof(double), n, f); fclose(f); }
    return 0;
}
