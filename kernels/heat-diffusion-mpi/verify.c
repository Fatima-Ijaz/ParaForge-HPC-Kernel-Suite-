/*
 * Small helper that compares two grids written by ./serial and ./parallel.
 * Both files are raw N*N doubles. It prints the largest absolute difference and
 * exits non-zero if that difference is above 1e-12 (so `make check` can stop on it).
 * Because the MPI code does the same arithmetic in the same order as the serial code,
 * the difference is normally exactly 0.
 *
 * Usage: ./verify a.bin b.bin N
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s a.bin b.bin N\n", argv[0]); return 2; }
    size_t n = (size_t)atoi(argv[3]) * atoi(argv[3]);
    double *a = malloc(n * sizeof(double)), *b = malloc(n * sizeof(double));
    FILE *fa = fopen(argv[1], "rb"), *fb = fopen(argv[2], "rb");
    if (!fa || !fb || fread(a, sizeof(double), n, fa) != n || fread(b, sizeof(double), n, fb) != n) {
        fprintf(stderr, "read error\n"); return 2;
    }
    double md = 0.0;
    for (size_t k = 0; k < n; k++) md = fmax(md, fabs(a[k] - b[k]));
    printf("max_abs_diff=%.3e %s\n", md, md == 0.0 ? "PASS (bitwise identical)" : (md < 1e-12 ? "PASS (tol 1e-12)" : "FAIL"));
    return md < 1e-12 ? 0 : 1;
}
