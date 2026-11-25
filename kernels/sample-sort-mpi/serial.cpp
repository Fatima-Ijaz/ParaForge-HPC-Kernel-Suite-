// Sample-sort baseline: sort all N keys in a single process with std::sort.
//
// The keys come from the same generator the MPI program uses, so both sort the same data.
// Only the sort itself is timed. Afterwards the result is checked (sorted, and the multiset
// fingerprint matches the input), which is why the line ends in PASS or FAIL.
//
// Usage: ./serial N [u|z]    (N keys; u = uniform, z = Zipf-skewed with duplicates)
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "keys.h"

int main(int argc, char **argv) {
    uint64_t N = argc > 1 ? strtoull(argv[1], 0, 10) : 100000000ULL;
    char dist = argc > 2 ? argv[2][0] : 'u';
    std::vector<uint32_t> a(N);
    gen_keys(a.data(), 0, N, dist, 7);
    uint64_t f1 = 0, f2 = 0; fingerprint(a.data(), N, f1, f2);

    auto t0 = std::chrono::steady_clock::now();
    std::sort(a.begin(), a.end());
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    bool ok = std::is_sorted(a.begin(), a.end());
    uint64_t g1 = 0, g2 = 0; fingerprint(a.data(), N, g1, g2);
    ok = ok && f1 == g1 && f2 == g2;
    printf("serial N=%llu dist=%c time=%.6f fp=%016llx %s\n", (unsigned long long)N, dist, t,
           (unsigned long long)(g1 ^ g2), ok ? "PASS" : "FAIL");
    return !ok;
}
