# Contributions

Two people, five kernels. Each kernel has one **lead** (writes most of the code and the matching
report section) and one **reviewer** (reads it, builds it from scratch and re-runs the checks).
Whoever does not lead a kernel reviews it.

| Kernel | Lead | Reviewer |
|---|---|---|
| Heat diffusion (MPI) | Zahra Amjad | Fatima Ijaz |
| k-mer counting (OpenMP) | Zahra Amjad | Fatima Ijaz |
| Monte Carlo pricing (OpenMP + MPI) | Zahra Amjad | Fatima Ijaz |
| PageRank (OpenMP) | Fatima Ijaz | Zahra Amjad |
| Sample sort (MPI) | Fatima Ijaz | Zahra Amjad |

Shared by both: the slide deck, the cross-kernel comparison, the benchmark methodology and the
final from-scratch build on a clean Linux install.

## Zahra Amjad (23L-3072)

* **Leads:** heat diffusion, k-mer counting, Monte Carlo pricing
* **Work:** halo-exchange and Cartesian-topology code, blocking and non-blocking variants,
  convergence-check experiments; racy / atomic / private / reduction histogram versions and the
  false-sharing test; xoshiro256** stream jumping, reproducible reductions and the MPI variant
  of the pricer
* **Files:** `kernels/heat-diffusion-mpi/`, `kernels/kmer-counting-openmp/`,
  `kernels/monte-carlo-pricing/`
* **Reviewed:** PageRank, sample sort
* **Presented:** heat diffusion, k-mer counting, Monte Carlo
* **Approx. hours:** _to be filled in by Zahra_

## Fatima Ijaz (23L-3042)

* **Leads:** PageRank, sample sort
* **Work:** R-MAT generator and CSR construction, pull and push kernels, schedule and chunk-size
  sweep, imbalance and memory-traffic analysis; distributed sample sort, `Alltoallv`
  redistribution, verification, ping-pong benchmark and communication cost model; repository
  structure, build tooling and CI
* **Files:** `kernels/pagerank-openmp/`, `kernels/sample-sort-mpi/`, `.github/`, top-level `Makefile`
* **Reviewed:** heat diffusion, k-mer counting, Monte Carlo
* **Presented:** PageRank, sample sort
* **Approx. hours:** _to be filled in by Fatima_

## Working agreement

* Every kernel is built and checked (`make check`) by its reviewer before it is considered done.
* Benchmarks are run with the same script on an otherwise idle machine, three repetitions, median reported.
* The git history is the authoritative record of who changed what; this file only summarises it.
