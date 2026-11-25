#!/bin/bash
# Monte Carlo pricing benchmark driver -> ../../results/cs3_raw.txt
#
# Covers strong scaling with correct streams, the shared-generator-with-lock experiment, the
# same-seed experiment (20 trials, to compare the reported standard error with the observed
# spread), a fixed-time-budget comparison, reproducibility across worker counts, and the MPI
# version of the same estimator.
set -e
cd "$(dirname "$0")"
make -s all
REPS=${REPS:-3}
M=${M:-100000000}
TLIST=${TLIST:-"1 2 4"}
export OMP_PROC_BIND=close OMP_PLACES=cores
OUT=../../results/cs3_raw.txt
: > $OUT

# strong scaling (good streams)
for r in $(seq $REPS); do ./serial $M >> $OUT; done
for t in $TLIST; do for r in $(seq $REPS); do OMP_NUM_THREADS=$t ./parallel $M 2026 good >> $OUT; done; done

# shared generator behind a lock: serialization (smaller M, it is slow)
for t in 1 2; do for r in $(seq $REPS); do OMP_NUM_THREADS=$t ./parallel 10000000 2026 shared >> $OUT; done; done

# same seed vs. independent streams: 20 independent trials with p=8 workers, M=1e7 each
# (p=8 here is about statistics, not speed, so oversubscription does not matter)
for s in $(seq 1 20); do
  echo "TRIAL $(OMP_NUM_THREADS=8 ./parallel 10000000 $s good)" >> $OUT
  echo "TRIAL $(OMP_NUM_THREADS=8 ./parallel 10000000 $s sameseed)" >> $OUT
done

# fixed wall-clock budget: p=2 processes 2x the samples in the same time
for r in $(seq $REPS); do echo "FIXED $(OMP_NUM_THREADS=2 ./parallel $((2*M)) 2026 good)" >> $OUT; done

# reproducibility: same seed, varying p, repeated
for t in 1 2 3 4 8; do for r in 1 2; do echo "REPRO $(OMP_NUM_THREADS=$t ./parallel 10000000 2026 good)" >> $OUT; done; done

# bonus: MPI version of the same estimator
for t in 1 2; do for r in $(seq $REPS); do echo "MPI $(mpirun --oversubscribe -np $t ./parallel_mpi $M 2026)" >> $OUT; done; done
echo "done -> $OUT"
