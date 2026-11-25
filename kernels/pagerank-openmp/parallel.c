/*
 * PageRank - OpenMP version with pull and push formulations and selectable loop schedule.
 *
 * Same maths as serial.c; what changes is how the per-iteration edge loop is parallelised.
 *
 *   pull       Each thread owns a set of DESTINATION vertices and sums the contributions of
 *              their in-neighbours. Every r_new[v] has exactly one writer, so no locking is
 *              needed; the cost is random reads of c[u].
 *   push       Each thread owns a set of SOURCE vertices and adds r[u]/outdeg(u) into each
 *              out-neighbour. Different threads can hit the same destination at the same
 *              time, so each update is an `omp atomic` (a compare-and-swap loop for doubles).
 *   pushplain  push without atomics. It is only correct with one thread and exists purely to
 *              measure how much the atomics cost by themselves.
 *
 * Load balance: the graph is skewed, so equal numbers of vertices do not mean equal numbers
 * of edges. The vertex loop uses schedule(runtime), set from the command line through
 * omp_set_schedule(), so static / dynamic / guided and different chunk sizes can be compared
 * without recompiling. The time each thread spends in the edge loop is recorded and reported
 * as R_time = slowest thread / mean thread (1.0 = perfectly balanced). R_edges_static is the
 * same ratio computed from edge counts of equal static vertex blocks.
 *
 * Usage: OMP_NUM_THREADS=p ./parallel scale edgefactor MODE SCHED CHUNK [ref.bin]
 *   MODE   pull | push | pushplain
 *   SCHED  static | dynamic | guided        CHUNK  chunk size (0 = OpenMP default)
 *   ref.bin ranks saved by ./serial; if given, max |diff| and L1 diff are checked (< 1e-9)
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "graph.h"

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s scale ef pull|push|pushplain static|dynamic|guided chunk [ref.bin]\n", argv[0]); return 1; }
    int scale = atoi(argv[1]), ef = atoi(argv[2]);
    const char *mode = argv[3], *sched = argv[4];
    int chunk = atoi(argv[5]);
    const char *ref = argc > 6 ? argv[6] : NULL;
    omp_sched_t kind = !strcmp(sched, "dynamic") ? omp_sched_dynamic : !strcmp(sched, "guided") ? omp_sched_guided : omp_sched_static;
    omp_set_schedule(kind, chunk);   /* picked up by every `schedule(runtime)` loop below */
    int P = omp_get_max_threads();
    int push = !strncmp(mode, "push", 4), plain = !strcmp(mode, "pushplain");

    graph_t g = make_rmat(scale, ef, 42);
    int n = g.n;
    double *r = malloc(n * sizeof(double)), *rn = malloc(n * sizeof(double)), *c = malloc(n * sizeof(double));
    double *tthr = calloc(P, sizeof(double));   /* accumulated edge-loop time per thread */
    /* initialise in parallel so memory is first-touched by the threads that will use it */
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < n; v++) { r[v] = 1.0 / n; rn[v] = 0.0; c[v] = 0.0; }

    int it; double diff = 0.0;
    double t0 = now_sec();
    for (it = 1; it <= MAXIT; it++) {
        /* per-vertex contribution and dangling total; cheap and regular, so plain static */
        double dang = 0.0;
        #pragma omp parallel for schedule(static) reduction(+ : dang)
        for (int u = 0; u < n; u++) {
            if (g.outdeg[u]) c[u] = r[u] / g.outdeg[u]; else { c[u] = 0.0; dang += r[u]; }
        }
        double base = (1.0 - DAMP) / n + DAMP * dang / n;
        diff = 0.0;
        if (!push) {
            /* ---- pull: one writer per destination vertex, so no atomics ---- */
            #pragma omp parallel reduction(+ : diff)
            {
                double ts = now_sec();
                #pragma omp for schedule(runtime) nowait
                for (int v = 0; v < n; v++) {
                    double s = 0.0;
                    for (long e = g.in_off[v]; e < g.in_off[v + 1]; e++) s += c[g.in_src[e]];
                    rn[v] = base + DAMP * s;
                    diff += fabs(rn[v] - r[v]);
                }
                tthr[omp_get_thread_num()] += now_sec() - ts;
            }
        } else {
            /* ---- push: accumulate into rn, so it has to be cleared first ---- */
            #pragma omp parallel for schedule(static)
            for (int v = 0; v < n; v++) rn[v] = 0.0;
            #pragma omp parallel
            {
                double ts = now_sec();
                if (plain) {
                    #pragma omp for schedule(runtime) nowait
                    for (int u = 0; u < n; u++) {
                        double cu = c[u];
                        for (long e = g.out_off[u]; e < g.out_off[u + 1]; e++) rn[g.out_dst[e]] += cu;
                    }
                } else {
                    #pragma omp for schedule(runtime) nowait
                    for (int u = 0; u < n; u++) {
                        double cu = c[u];
                        for (long e = g.out_off[u]; e < g.out_off[u + 1]; e++) {
                            #pragma omp atomic
                            rn[g.out_dst[e]] += cu;
                        }
                    }
                }
                tthr[omp_get_thread_num()] += now_sec() - ts;
            }
            #pragma omp parallel for schedule(static) reduction(+ : diff)
            for (int v = 0; v < n; v++) { rn[v] = base + DAMP * rn[v]; diff += fabs(rn[v] - r[v]); }
        }
        double *t = r; r = rn; rn = t;
        if (diff < EPS) break;
    }
    double t = now_sec() - t0;

    /* load imbalance of the edge loop: measured per-thread time, and the edge-count imbalance
       that equal-sized static vertex blocks would have (a property of the graph alone) */
    double tmax = 0, tsum = 0;
    for (int q = 0; q < P; q++) { tsum += tthr[q]; if (tthr[q] > tmax) tmax = tthr[q]; }
    double R_time = tmax / (tsum / P);
    long emax = 0;
    for (int q = 0; q < P; q++) {
        int lo = (int)((long)n * q / P), hi = (int)((long)n * (q + 1) / P);
        long e = push ? g.out_off[hi] - g.out_off[lo] : g.in_off[hi] - g.in_off[lo];
        if (e > emax) emax = e;
    }
    double R_edges = (double)emax / ((double)g.m / P);

    double maxd = 0.0, l1 = 0.0;
    if (ref) {
        double *rr = malloc(n * sizeof(double));
        FILE *f = fopen(ref, "rb");
        if (f && fread(rr, sizeof(double), n, f) == (size_t)n) {
            for (int v = 0; v < n; v++) { double d = fabs(rr[v] - r[v]); l1 += d; if (d > maxd) maxd = d; }
        } else maxd = l1 = -1;
        if (f) fclose(f);
        free(rr);
    }
    printf("parallel scale=%d ef=%d m=%ld mode=%s sched=%s chunk=%d threads=%d iters=%d time=%.6f time_per_iter=%.6f R_time=%.3f R_edges_static=%.3f Medges_per_s=%.1f ref_maxdiff=%.3e ref_l1=%.3e %s\n",
           scale, ef, g.m, mode, sched, chunk, P, it, t, t / it, R_time, R_edges, (double)g.m * it / t / 1e6,
           maxd, l1, !ref ? "" : (maxd >= 0 && l1 < 1e-9) ? "PASS" : "FAIL");
    return 0;
}
