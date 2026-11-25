// Distributed sample sort with MPI.
//
// N 32-bit keys are spread over p ranks and no rank ever holds the whole data set. After the
// run, rank 0 holds the smallest keys, rank 1 the next ones and so on, each rank's block
// sorted. The algorithm, step by step:
//
//   (1) local sort      every rank sorts its own N/p keys with std::sort
//   (2) sampling        every rank picks os*p-1 evenly spaced samples from its sorted data;
//                       MPI_Gather sends them to rank 0, which sorts them and chooses p-1
//                       splitters; MPI_Bcast gives the splitters to everybody
//   (3) partition       each rank cuts its sorted array into p pieces at the splitters
//                       (binary search with upper_bound, so equal keys stay together)
//   (4) redistribution  MPI_Alltoall tells every rank how many keys to expect from each
//                       peer, then MPI_Alltoallv moves the keys: piece j goes to rank j
//   (5) merge           the p sorted runs a rank received are merged pairwise into one
//
//        rank 0: [ sorted ]  --\                         /--> [ keys <= s1        ]
//        rank 1: [ sorted ]  ---+-- splitters s1..s(p-1) +---> [ s1 < keys <= s2   ]
//        rank 2: [ sorted ]  --/    + all-to-all exchange \--> [ keys > s2          ]
//
// Usage: mpirun -np p ./parallel N [u|z] [oversample]
//   N          total number of keys
//   u|z        uniform keys or Zipf-skewed keys with many duplicates
//   oversample os: more samples per rank give better splitters (default 1)
//
// Timing: barriers separate the phases, so the max-over-ranks time of each phase adds up to
// the total. Verification (not timed): each rank is sorted, the last key of a rank is not
// larger than the first key of the next non-empty rank, the total count is N, and the
// multiset fingerprint is unchanged. Load balance is reported as max / mean keys per rank.
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "keys.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    uint64_t N = argc > 1 ? strtoull(argv[1], 0, 10) : 100000000ULL;
    char dist = argc > 2 ? argv[2][0] : 'u';
    int os = argc > 3 ? atoi(argv[3]) : 1;

    /* this rank's slice of the global index range; keys are generated locally, nothing is read or sent */
    uint64_t lo = N * rank / p, hi = N * (rank + 1) / p;
    size_t n = hi - lo;
    std::vector<uint32_t> a(n);
    gen_keys(a.data(), lo, hi, dist, 7);
    /* fingerprint of the input, summed over all ranks, to compare with the output later */
    uint64_t f[2] = {0, 0}, F0[2];
    fingerprint(a.data(), n, f[0], f[1]);
    MPI_Allreduce(f, F0, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    double tp[5] = {0}, t;
    MPI_Barrier(MPI_COMM_WORLD);
    double tstart = MPI_Wtime();

    // (1) local sort
    t = MPI_Wtime();
    std::sort(a.begin(), a.end());
    MPI_Barrier(MPI_COMM_WORLD); tp[0] = MPI_Wtime() - t;

    // (2) sampling and splitters, (3) partition
    // samples are taken at evenly spaced positions of the already sorted local array, which
    // makes them "regular samples" - a good picture of the local key distribution
    t = MPI_Wtime();
    int ns = os * p - 1;
    std::vector<uint32_t> samp(ns), all, split(p > 1 ? p - 1 : 0);
    for (int i = 0; i < ns; i++) samp[i] = n ? a[(size_t)((i + 1) * (double)n / (ns + 1))] : 0;
    if (rank == 0) all.resize((size_t)ns * p);
    MPI_Gather(samp.data(), ns, MPI_UINT32_T, all.data(), ns, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::sort(all.begin(), all.end());
        for (int j = 1; j < p; j++) split[j - 1] = all[(size_t)j * all.size() / p];
    }
    if (p > 1) MPI_Bcast(split.data(), p - 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<int> scnt(p), sdsp(p), rcnt(p), rdsp(p);
    size_t prev = 0;
    // scnt[j]/sdsp[j] = how many keys go to rank j and where they start in the sorted array
    for (int j = 0; j < p; j++) {
        size_t end = j < p - 1 ? std::upper_bound(a.begin() + prev, a.end(), split[j]) - a.begin() : n;
        scnt[j] = (int)(end - prev); sdsp[j] = (int)prev; prev = end;
    }
    MPI_Barrier(MPI_COMM_WORLD); tp[1] = MPI_Wtime() - t;

    // (4) redistribution
    t = MPI_Wtime();
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    size_t m = 0;
    for (int j = 0; j < p; j++) { rdsp[j] = (int)m; m += rcnt[j]; }
    // receive buffer: new[] without () leaves it uninitialised, so there is no zero-fill pass
    std::unique_ptr<uint32_t[]> b(new uint32_t[m ? m : 1]);
    double tc = MPI_Wtime();
    MPI_Alltoallv(a.data(), scnt.data(), sdsp.data(), MPI_UINT32_T,
                  b.get(), rcnt.data(), rdsp.data(), MPI_UINT32_T, MPI_COMM_WORLD);
    tp[4] = MPI_Wtime() - tc;                     // the Alltoallv call alone
    MPI_Barrier(MPI_COMM_WORLD); tp[2] = MPI_Wtime() - t;
    std::vector<uint32_t>().swap(a);              // input shard no longer needed

    // (5) merge p sorted runs
    // runs[] holds the start offsets of the received runs; each pass merges neighbouring pairs
    // into tmp and then swaps the buffers, so log2(p) passes leave one sorted run
    t = MPI_Wtime();
    std::vector<size_t> runs(p + 1);
    for (int j = 0; j < p; j++) runs[j] = rdsp[j];
    runs[p] = m;
    std::unique_ptr<uint32_t[]> tmp(p > 1 ? new uint32_t[m ? m : 1] : nullptr);
    while (runs.size() > 2) {
        std::vector<size_t> nr; nr.push_back(0);
        for (size_t r = 0; r + 1 < runs.size(); r += 2) {
            size_t s0 = runs[r], s1 = runs[r + 1], s2 = r + 2 < runs.size() ? runs[r + 2] : runs[r + 1];
            std::merge(b.get() + s0, b.get() + s1, b.get() + s1, b.get() + s2, tmp.get() + s0);
            nr.push_back(s2);
        }
        b.swap(tmp); runs.swap(nr);
    }
    MPI_Barrier(MPI_COMM_WORLD); tp[3] = MPI_Wtime() - t;
    double total = MPI_Wtime() - tstart;

    // ---- verification (not timed) ----
    int ok = std::is_sorted(b.get(), b.get() + m);
    f[0] = f[1] = 0;
    fingerprint(b.get(), m, f[0], f[1]);
    uint64_t F1[2]; MPI_Reduce(f, F1, 2, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    long long info[3] = {(long long)m, m ? (long long)b[0] : -1, m ? (long long)b[m - 1] : -1};
    std::vector<long long> ginfo(3 * p);
    MPI_Gather(info, 3, MPI_LONG_LONG, ginfo.data(), 3, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    int allok; MPI_Reduce(&ok, &allok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    double tmx[5]; MPI_Reduce(tp, tmx, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double tot; MPI_Reduce(&total, &tot, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long cnt = 0, mx = 0, lastmax = -1; bool order = true;
        std::string counts;
        for (int r = 0; r < p; r++) {
            long long c = ginfo[3 * r];
            cnt += c; if (c > mx) mx = c;
            counts += std::to_string(c) + (r + 1 < p ? "," : "");
            if (c) { if (ginfo[3 * r + 1] < lastmax) order = false; lastmax = ginfo[3 * r + 2]; }
        }
        bool pass = allok && order && cnt == (long long)N && F0[0] == F1[0] && F0[1] == F1[1];
        printf("parallel N=%llu dist=%c os=%d p=%d time=%.6f local_sort=%.6f sampling=%.6f alltoallv=%.6f merge=%.6f a2av_call=%.6f "
               "max_over_mean=%.4f counts=%s %s\n",
               (unsigned long long)N, dist, os, p, tot, tmx[0], tmx[1], tmx[2], tmx[3], tmx[4],
               (double)mx / ((double)N / p), counts.c_str(), pass ? "PASS" : "FAIL");
    }
    MPI_Finalize();
    return 0;
}
