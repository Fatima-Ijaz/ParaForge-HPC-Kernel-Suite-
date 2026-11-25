/*
 * Heat diffusion on a square plate - MPI version (domain decomposition).
 *
 * Same numerical scheme as serial.c, but the N x N grid is cut into blocks and each
 * MPI rank owns one block. A rank needs the cells just outside its block to update
 * the cells on its edge, so every rank stores a one-cell "ghost frame" around its
 * block and refreshes it from its neighbours before each time step (halo exchange).
 *
 *      rank layout (2-D example, 4 ranks)         local array of one rank
 *      +---------+---------+                      g g g g g g
 *      | (0,0)   | (0,1)   |                      g o o o o g     o = owned cells
 *      +---------+---------+                      g o o o o g     g = ghost cells
 *      | (1,0)   | (1,1)   |                      g o o o o g       (copies of the
 *      +---------+---------+                      g g g g g g        neighbour's edge)
 *
 * Two decompositions are available, both built on MPI_Cart_create:
 *   1-D  P x 1 grid of ranks, each owns a band of full rows (2 halos of N values)
 *   2-D  grid from MPI_Dims_create, each owns a rectangle (4 halos of about N/sqrt(P))
 * and two ways of doing the exchange:
 *   b    blocking, 4 x MPI_Sendrecv, then update everything
 *   nb   non-blocking, post Irecv/Isend, update the interior (which does not need
 *        ghost values) while the messages are in flight, Waitall, then update the
 *        thin frame of cells next to the ghosts
 *
 * Usage: mpirun -np P ./parallel N T K DECOMP MODE [outfile]
 *   DECOMP  1 = row bands, 2 = 2-D blocks
 *   MODE    b = blocking, nb = non-blocking with overlap
 *   K       global max-change check with MPI_Allreduce every K steps (0 = never)
 *   outfile optional; rank 0 gathers the final grid and writes raw doubles (not timed)
 *
 * Printed timings are the maximum over ranks: `time` is the whole loop, `comm` the time
 * spent in halo-exchange calls, `allreduce` the time spent in the convergence check.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ALPHA 0.2
#define IDX(i, j) ((size_t)(i) * (lm + 2) + (j))

/* N       global grid size
 * ln, lm  rows / columns owned by this rank
 * gr0,gc0 global row / column of this rank's first owned cell */
static int N, ln, lm, gr0, gc0;
static int ilo, ihi, jlo, jhi;      /* local index range of cells that are updated */

static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }

/* Split n items over p parts as evenly as possible; part r gets [*start, *start + *len).
 * The first (n % p) parts get one extra item, so N does not have to divide evenly. */
static void split(int n, int p, int r, int *start, int *len) {
    int q = n / p, rem = n % p;
    *len = q + (r < rem);
    *start = r * q + imin(r, rem);
}

/* Apply the stencil to local rows i0..i1 and columns j0..j1, reading u and writing v.
 * The range is clamped to the cells that really get updated (the global boundary is
 * fixed). Returns the largest change seen, but only if `check` is set. */
static double update(const double *u, double *v, int i0, int i1, int j0, int j1, int check) {
    i0 = imax(i0, ilo); i1 = imin(i1, ihi);
    j0 = imax(j0, jlo); j1 = imin(j1, jhi);
    double md = 0.0;
    for (int i = i0; i <= i1; i++) {
        const double *up = u + IDX(i - 1, 0), *uc = u + IDX(i, 0), *dn = u + IDX(i + 1, 0);
        double *vc = v + IDX(i, 0);
        for (int j = j0; j <= j1; j++) {
            double nv = uc[j] + ALPHA * (up[j] + dn[j] + uc[j - 1] + uc[j + 1] - 4.0 * uc[j]);
            vc[j] = nv;
            if (check) { double d = fabs(nv - uc[j]); if (d > md) md = d; }
        }
    }
    return md;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 6) {
        if (!rank) fprintf(stderr, "usage: %s N T K DECOMP(1|2) MODE(b|nb) [outfile]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    N = atoi(argv[1]);
    int T = atoi(argv[2]), K = atoi(argv[3]), decomp = atoi(argv[4]);
    int nonblocking = (strcmp(argv[5], "nb") == 0);
    const char *out = argc > 6 ? argv[6] : NULL;

    /* --- Cartesian topology ------------------------------------------------ */
    /* non-periodic grid: ranks at the plate edge get MPI_PROC_NULL as a neighbour, so the
     * halo calls to "outside" are no-ops and no special-casing is needed below */
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
    if (decomp == 1) { dims[0] = p; dims[1] = 1; }
    else MPI_Dims_create(p, 2, dims);
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 2, coords);
    int north, south, west, east;
    MPI_Cart_shift(cart, 0, 1, &north, &south);
    MPI_Cart_shift(cart, 1, 1, &west, &east);

    split(N, dims[0], coords[0], &gr0, &ln);
    split(N, dims[1], coords[1], &gc0, &lm);
    if (ln < 2 || lm < 2) { if (!rank) fprintf(stderr, "block too small\n"); MPI_Abort(cart, 1); }

    /* local row/column 1 is the first owned cell; the global boundary (fixed) starts
     * at global index 0 and N-1, so the first and last owned cells there are skipped */
    ilo = (gr0 == 0) ? 2 : 1;
    ihi = (gr0 + ln == N) ? ln - 1 : ln;
    jlo = (gc0 == 0) ? 2 : 1;
    jhi = (gc0 + lm == N) ? lm - 1 : lm;

    size_t tot = (size_t)(ln + 2) * (lm + 2);
    double *u = calloc(tot, sizeof(double)), *v = calloc(tot, sizeof(double));
    if (gr0 == 0) for (int j = 1; j <= lm; j++) { u[IDX(1, j)] = 100.0; v[IDX(1, j)] = 100.0; }

    /* a row is contiguous in memory, a column is strided; describe both as datatypes so
     * the exchange calls stay one-liners */
    MPI_Datatype rowT, colT;
    MPI_Type_contiguous(lm, MPI_DOUBLE, &rowT);           MPI_Type_commit(&rowT);
    MPI_Type_vector(ln, 1, lm + 2, MPI_DOUBLE, &colT);    MPI_Type_commit(&colT);

    double t_comm = 0.0, t_red = 0.0, gmax = 0.0;
    MPI_Barrier(cart);
    double t0 = MPI_Wtime();

    for (int t = 1; t <= T; t++) {
        int check = (K > 0 && t % K == 0);
        double md;
        if (!nonblocking) {
            /* blocking: each Sendrecv sends my edge one way while receiving the neighbour's
             * edge from the other way, which cannot deadlock */
            double c0 = MPI_Wtime();
            MPI_Sendrecv(&u[IDX(1, 1)], 1, rowT, north, 0, &u[IDX(ln + 1, 1)], 1, rowT, south, 0, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&u[IDX(ln, 1)], 1, rowT, south, 1, &u[IDX(0, 1)], 1, rowT, north, 1, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&u[IDX(1, 1)], 1, colT, west, 2, &u[IDX(1, lm + 1)], 1, colT, east, 2, cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&u[IDX(1, lm)], 1, colT, east, 3, &u[IDX(1, 0)], 1, colT, west, 3, cart, MPI_STATUS_IGNORE);
            t_comm += MPI_Wtime() - c0;
            md = update(u, v, 1, ln, 1, lm, check);
        } else {
            MPI_Request rq[8];
            double c0 = MPI_Wtime();
            MPI_Irecv(&u[IDX(ln + 1, 1)], 1, rowT, south, 0, cart, &rq[0]);
            MPI_Irecv(&u[IDX(0, 1)],      1, rowT, north, 1, cart, &rq[1]);
            MPI_Irecv(&u[IDX(1, lm + 1)], 1, colT, east,  2, cart, &rq[2]);
            MPI_Irecv(&u[IDX(1, 0)],      1, colT, west,  3, cart, &rq[3]);
            MPI_Isend(&u[IDX(1, 1)],      1, rowT, north, 0, cart, &rq[4]);
            MPI_Isend(&u[IDX(ln, 1)],     1, rowT, south, 1, cart, &rq[5]);
            MPI_Isend(&u[IDX(1, 1)],      1, colT, west,  2, cart, &rq[6]);
            MPI_Isend(&u[IDX(1, lm)],     1, colT, east,  3, cart, &rq[7]);
            t_comm += MPI_Wtime() - c0;
            /* the interior (everything except the outermost owned ring) never reads the
             * ghost cells, so it can be computed while the messages are still moving */
            md = update(u, v, 2, ln - 1, 2, lm - 1, check);
            c0 = MPI_Wtime();
            MPI_Waitall(8, rq, MPI_STATUSES_IGNORE);
            t_comm += MPI_Wtime() - c0;
            /* now the ghosts have arrived: finish the ring of owned cells next to them
             * (top row, bottom row, then the left and right columns without the corners) */
            double m1 = update(u, v, 1, 1, 1, lm, check);
            double m2 = update(u, v, ln, ln, 1, lm, check);
            double m3 = update(u, v, 2, ln - 1, 1, 1, check);
            double m4 = update(u, v, 2, ln - 1, lm, lm, check);
            md = fmax(fmax(md, m1), fmax(m2, fmax(m3, m4)));
        }
        if (check) {
            double c0 = MPI_Wtime();
            MPI_Allreduce(&md, &gmax, 1, MPI_DOUBLE, MPI_MAX, cart);
            t_red += MPI_Wtime() - c0;
        }
        double *tmp = u; u = v; v = tmp;
    }

    MPI_Barrier(cart);
    double t1 = MPI_Wtime();
    double el = t1 - t0, mx[3], loc[3] = {el, t_comm, t_red};
    MPI_Reduce(loc, mx, 3, MPI_DOUBLE, MPI_MAX, 0, cart);

    double lsum = 0.0, gsum;
    for (int i = 1; i <= ln; i++) for (int j = 1; j <= lm; j++) lsum += u[IDX(i, j)];
    MPI_Reduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, cart);

    if (!rank)
        printf("parallel N=%d T=%d K=%d p=%d decomp=%d dims=%dx%d mode=%s time=%.6f comm=%.6f allreduce=%.6f checksum=%.12e maxdiff=%.6e\n",
               N, T, K, p, decomp, dims[0], dims[1], nonblocking ? "nb" : "b", mx[0], mx[1], mx[2], gsum, gmax);

    /* --- optional gather for verification (not timed) ---------------------- */
    /* rank 0 collects every block and stitches them into one N x N array in row-major
     * order, which is the same file layout serial.c writes */
    if (out) {
        double *blk = malloc((size_t)ln * lm * sizeof(double));
        for (int i = 0; i < ln; i++) memcpy(blk + (size_t)i * lm, &u[IDX(i + 1, 1)], lm * sizeof(double));
        if (rank == 0) {
            double *g = malloc((size_t)N * N * sizeof(double));
            for (int r = 0; r < p; r++) {
                int c[2], r0, rl, c0, cl;
                MPI_Cart_coords(cart, r, 2, c);
                split(N, dims[0], c[0], &r0, &rl);
                split(N, dims[1], c[1], &c0, &cl);
                double *b = blk;
                if (r) { b = malloc((size_t)rl * cl * sizeof(double)); MPI_Recv(b, rl * cl, MPI_DOUBLE, r, 9, cart, MPI_STATUS_IGNORE); }
                for (int i = 0; i < rl; i++) memcpy(g + (size_t)(r0 + i) * N + c0, b + (size_t)i * cl, cl * sizeof(double));
                if (r) free(b);
            }
            FILE *f = fopen(out, "wb"); fwrite(g, sizeof(double), (size_t)N * N, f); fclose(f);
            free(g);
        } else {
            MPI_Send(blk, ln * lm, MPI_DOUBLE, 0, 9, cart);
        }
        free(blk);
    }

    MPI_Type_free(&rowT); MPI_Type_free(&colT);
    free(u); free(v);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
