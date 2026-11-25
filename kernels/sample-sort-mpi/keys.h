// Key generation for the sample-sort benchmark.
//
// Key number i is a pure function of (seed, i). Because of that, every MPI rank can create
// its own shard of the data independently (rank r simply handles indices N*r/p .. N*(r+1)/p),
// nobody ever needs the whole array, and the complete data set is identical for every number
// of ranks. The single-process std::sort baseline therefore sorts exactly the same keys.
//
// Two distributions are available:
//   'u'  uniform 32-bit keys (balanced buckets are easy)
//   'z'  Zipf-like skew over 1024 distinct values, so there are lots of duplicates and some
//        values are very frequent (this is what stresses the splitter selection)
#ifndef KEYS_H
#define KEYS_H
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

static inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Zipf(s = 1.0) over 1024 distinct values: the most frequent value is ~13.4% of all keys.
// The cumulative distribution is tabulated once; a key is drawn by binary-searching a
// uniform random number in that table.
struct ZipfTable {
    std::vector<double> cdf;
    ZipfTable() : cdf(1024) {
        double acc = 0;
        for (int k = 0; k < 1024; k++) { acc += 1.0 / (k + 1); cdf[k] = acc; }
        for (auto &c : cdf) c /= acc;
    }
    uint32_t key(uint64_t r) const {
        double u = ((r >> 11) + 0.5) * 0x1.0p-53;
        int k = (int)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        return (uint32_t)k * 4000037u;              // spread the 1024 values over 32-bit range
    }
};

// Fill out[0 .. hi-lo) with the keys of global indices [lo, hi).
// The seed is mixed with the index through a 64-bit finalizer (splitmix64 style), which
// behaves like an independent random draw per index.
static void gen_keys(uint32_t *out, uint64_t lo, uint64_t hi, char dist, uint64_t seed) {
    static const ZipfTable Z;
    for (uint64_t i = lo; i < hi; i++) {
        uint64_t r = mix64(seed * 0x100000001B3ULL + i);
        out[i - lo] = dist == 'z' ? Z.key(r) : (uint32_t)(r >> 32);
    }
}

// Order-independent fingerprint of a multiset: sum of the keys and sum of hashed keys, both
// mod 2^64. If a sort loses, duplicates or alters a key, at least one of the sums changes,
// and adding the per-rank fingerprints before and after the sort needs no communication of
// the data itself.
static inline void fingerprint(const uint32_t *a, size_t n, uint64_t &s1, uint64_t &s2) {
    for (size_t i = 0; i < n; i++) { s1 += a[i]; s2 += mix64(a[i]); }
}

#endif
