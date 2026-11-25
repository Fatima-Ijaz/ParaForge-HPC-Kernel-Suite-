/*
 * Heat diffusion on a square plate - serial reference version.
 *
 * This is the baseline that every MPI run is timed against and checked against.
 * It solves the 2-D heat equation with a plain explicit finite-difference scheme:
 * each cell moves a little towards the average of its four neighbours,
 *
 *     next[i][j] = cur[i][j] + ALPHA * (N + S + W + E - 4 * cur[i][j])
 *
 * The plate is N x N cells including its edge. The top edge is held at 100 degrees
 * and the other three edges at 0; those edge cells are fixed and never updated.
 * Two buffers (current / next) are swapped after every step, so a step only ever
 * reads values written by the previous step.
 *
 * Usage: ./serial N T K [outfile]
 *   N        grid size (N x N)
 *   T        number of time steps
 *   K        measure the largest per-cell change every K steps (0 = never).
 *            The measurement adds a fabs() and a compare to the inner loop, so it
 *            is only done on every K-th step.
 *   outfile  optional; dump the final grid as raw doubles so ./verify can compare it
 *            with the output of the MPI version.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define ALPHA 0.2   /* diffusion coefficient; the explicit scheme is stable for ALPHA <= 0.25 */

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    int T = argc > 2 ? atoi(argv[2]) : 500;
    int K = argc > 3 ? atoi(argv[3]) : 50;
    const char *out = argc > 4 ? argv[4] : NULL;

    double *u = calloc((size_t)N * N, sizeof(double));
    double *v = calloc((size_t)N * N, sizeof(double));
    /* hot top edge; it has to be set in both buffers because the edge is never rewritten */
    for (int j = 0; j < N; j++) { u[j] = 100.0; v[j] = 100.0; }

    double maxdiff = 0.0;
    double t0 = now();
    for (int t = 1; t <= T; t++) {
        int check = (K > 0 && t % K == 0);
        double md = 0.0;
        /* interior cells only: row/column 0 and N-1 are the fixed boundary */
        for (int i = 1; i < N - 1; i++) {
            const double *up = u + (size_t)(i - 1) * N, *uc = u + (size_t)i * N, *dn = u + (size_t)(i + 1) * N;
            double *vc = v + (size_t)i * N;
            for (int j = 1; j < N - 1; j++) {
                double nv = uc[j] + ALPHA * (up[j] + dn[j] + uc[j - 1] + uc[j + 1] - 4.0 * uc[j]);
                vc[j] = nv;
                if (check) { double d = fabs(nv - uc[j]); if (d > md) md = d; }
            }
        }
        if (check) maxdiff = md;
        double *tmp = u; u = v; v = tmp;
    }
    double t1 = now();

    /* the checksum is a cheap fingerprint of the final field, handy for comparing runs */
    double sum = 0.0;
    for (size_t k = 0; k < (size_t)N * N; k++) sum += u[k];
    printf("serial N=%d T=%d K=%d time=%.6f checksum=%.12e maxdiff=%.6e\n", N, T, K, t1 - t0, sum, maxdiff);

    if (out) { FILE *f = fopen(out, "wb"); fwrite(u, sizeof(double), (size_t)N * N, f); fclose(f); }
    free(u); free(v);
    return 0;
}
