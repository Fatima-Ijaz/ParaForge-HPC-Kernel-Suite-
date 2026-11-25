#!/bin/bash
# k-mer counting benchmark driver -> ../../results/cs2_raw.txt
#
# For both the uniform and the skewed sequence and for k = 8 and k = 12 it runs the serial
# baseline and every OpenMP variant (A racy, B atomic, C private, R reduction) for the thread
# counts in TLIST. After that come two extra experiments: how many counts the racy version
# loses for small k, and the false-sharing test with and without padding.
# Threads are pinned (close / cores) to keep run-to-run noise down.
set -e
cd "$(dirname "$0")"
make -s all
REPS=${REPS:-3}
L=${L:-100000000}
TLIST=${TLIST:-"1 2 4"}
export OMP_PROC_BIND=close OMP_PLACES=cores
# version R privatises a 4^k array on each thread stack (64 MB at k=12): needs a large stack
ulimit -s unlimited; export OMP_STACKSIZE=256M
OUT=../../results/cs2_raw.txt
: > $OUT

for seq in u s; do for k in 8 12; do
  for r in $(seq $REPS); do ./serial $L $k $seq >> $OUT; done
  for v in A B C R; do for t in $TLIST; do
    for r in $(seq $REPS); do OMP_NUM_THREADS=$t ./parallel $L $k $seq $v >> $OUT; done
  done; done
done; done

# size of the data-race damage for small k (fewer bins -> more collisions on the same bin)
for k in 2 4 6; do for r in $(seq $REPS); do OMP_NUM_THREADS=2 ./parallel $L $k u A >> $OUT; done; done

# false sharing: k=1 -> 4 counters (16 B) per thread; pad 0 = adjacent, 16 = 64 B, 32 = 128 B
for t in 1 2; do for pad in 0 16 32; do
  for r in $(seq $REPS); do OMP_NUM_THREADS=$t ./parallel $L 1 u F $pad >> $OUT; done
done; done
for t in 1 2; do for r in $(seq $REPS); do OMP_NUM_THREADS=$t ./parallel $L 1 u C >> $OUT; done; done
echo "done -> $OUT"
