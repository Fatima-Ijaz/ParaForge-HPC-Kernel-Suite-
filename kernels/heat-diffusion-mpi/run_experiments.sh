#!/bin/bash
# Heat-diffusion benchmark driver.
#
# Runs the serial solver and every MPI variant (1-D / 2-D decomposition x blocking /
# non-blocking halo exchange) for the strong-scaling study, then the weak-scaling series
# and the convergence-check cost test. Each configuration is repeated REPS times and the
# raw output lines go to ../../results/cs1_raw.txt; results/analyze.py takes medians.
# Rank counts above the number of physical cores are launched with --oversubscribe and
# are marked as oversubscribed in the plots.
set -e
cd "$(dirname "$0")"
make -s all
REPS=${REPS:-3}
PLIST=${PLIST:-"1 2 4 8"}
MPIRUN="mpirun --oversubscribe --bind-to none"
OUT=../../results/cs1_raw.txt
: > $OUT

for N in 1024 4096; do
  for r in $(seq $REPS); do ./serial $N 500 50 >> $OUT; done
  for d in 1 2; do for m in b nb; do for p in $PLIST; do
    for r in $(seq $REPS); do $MPIRUN -np $p ./parallel $N 500 50 $d $m >> $OUT; done
  done; done; done
done

# weak scaling: keep about 1024^2 cells per rank (N grows like sqrt(p)), 2-D non-blocking
for pN in "1 1024" "2 1448" "4 2048" "8 2896"; do
  set -- $pN
  for r in $(seq $REPS); do
    echo "WEAK $($MPIRUN -np $1 ./parallel $2 500 50 2 nb)" >> $OUT
  done
done

# cost of the convergence check: every step (K=1), every 50 steps, and never (K=0)
for N in 1024 4096; do for p in 2 4; do for K in 0 1 50; do
  for r in $(seq $REPS); do echo "KTEST $($MPIRUN -np $p ./parallel $N 500 $K 2 nb)" >> $OUT; done
done; done; done
echo "done -> $OUT"
