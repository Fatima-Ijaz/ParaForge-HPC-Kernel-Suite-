#!/bin/bash
# PageRank benchmark driver -> ../../results/cs4_raw.txt and ../../results/cs4_degrees.txt
#
# Serial baseline (also writes the reference ranks and the degree histogram), then pull and
# push with static scheduling for the thread counts in TLIST, the cost of atomics (push with
# and without them on one thread), a schedule / chunk-size sweep, and the edge-count
# imbalance for larger thread counts.
set -e
cd "$(dirname "$0")"
make -s all
REPS=${REPS:-3}
S=${SCALE:-20}; E=${EF:-16}
TLIST=${TLIST:-"1 2 4"}
export OMP_PROC_BIND=close OMP_PLACES=cores
OUT=../../results/cs4_raw.txt
: > $OUT

./serial $S $E ref.bin ../../results/cs4_degrees.txt >> $OUT
for r in $(seq 2 $REPS); do ./serial $S $E >> $OUT; done

for mode in pull push; do for t in $TLIST; do for r in $(seq $REPS); do
  OMP_NUM_THREADS=$t ./parallel $S $E $mode static 0 ref.bin >> $OUT
done; done; done
for r in $(seq $REPS); do OMP_NUM_THREADS=1 ./parallel $S $E pushplain static 0 ref.bin >> $OUT; done

# schedule sweep at p = 2 (physical cores)
for mode in pull push; do
  for sc in "static 64" "static 1024" "dynamic 1" "dynamic 16" "dynamic 64" "dynamic 256" "dynamic 1024" "dynamic 4096" "dynamic 16384" "guided 0" "guided 64" "guided 1024"; do
    set -- $sc
    for r in $(seq $REPS); do OMP_NUM_THREADS=2 ./parallel $S $E $mode $1 $2 ref.bin >> $OUT; done
  done
done

# static edge-count imbalance for larger p (graph property; timings ignored, oversubscribed)
for t in 8 16; do echo "IMB $(OMP_NUM_THREADS=$t ./parallel $S $E pull static 0)" >> $OUT; done
rm -f ref.bin
echo "done -> $OUT"
