# Architecture and decomposition

This page shows, for each kernel, who owns which data and what is communicated. The source
files carry a header comment with the same information next to the code.

## Choosing a programming model

```mermaid
flowchart TD
    Q{Does the work share<br/>one big mutable structure?}
    Q -->|yes, and it fits on one node| OMP[OpenMP<br/>threads, shared arrays]
    Q -->|data is partitioned, communication<br/>pattern is regular| MPI[MPI<br/>ranks own blocks]
    OMP --> K2[k-mer histogram]
    OMP --> K4[PageRank graph]
    MPI --> K1[heat grid + halos]
    MPI --> K5[distributed keys]
    Q -->|independent samples,<br/>tiny reduction| BOTH[either model works]
    BOTH --> K3[Monte Carlo]
```

| Kernel | Model | Why |
|---|---|---|
| Heat diffusion | MPI | regular grid, neighbour-only communication, scales beyond one node |
| k-mer counting | OpenMP | one large shared histogram; the interesting question is how to update it |
| Monte Carlo | OpenMP and MPI | no sharing at all, so both fit; implemented twice to compare |
| PageRank | OpenMP | graph and rank vectors are shared and read randomly; partitioning a power-law graph for MPI is hard |
| Sample sort | MPI | data is distributed by definition; redistribution is an all-to-all |

## Heat diffusion: domain decomposition

![Heat decomposition](images/diag_cs1.png)

The hatched strips are the halo (ghost) cells. In the 1-D layout every interior rank swaps two
rows of N values per step; in the 2-D layout it swaps four edges of about N / sqrt(p) values.

## k-mer counting: chunk ownership and private histograms

![k-mer decomposition](images/diag_cs2.png)

## Monte Carlo: independent streams

![Monte Carlo streams](images/diag_cs3.png)

## PageRank: pull vs. push access patterns

![PageRank access patterns](images/diag_cs4.png)

## Sample sort: pipeline

![Sample sort pipeline](images/diag_cs5.png)

## Communication cost model used for sample sort

One-way message time is modelled as `T(m) = alpha + beta * m`, with alpha (latency) and beta
(seconds per byte) fitted from a ping-pong run. For the `Alltoallv` the model is extended with
a per-byte term gamma for first-touching the freshly allocated receive buffer:

```
T_alltoallv  ~  (p - 1) * alpha + beta * m_remote + gamma * m_recv
```

On the test machine this predicted 0.282 s against 0.291 s measured at p = 2.
