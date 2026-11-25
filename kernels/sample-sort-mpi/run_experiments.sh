#!/bin/bash
# Sample-sort benchmark driver -> ../../results/cs5_raw.txt and ../../results/cs5_pingpong.txt
#
# Ping-pong first (gives alpha and beta for the cost model), then the single-process std::sort
# baseline and the MPI sample sort for both key distributions and the rank counts in PLIST,
# and finally the oversampling sweep that shows how the number of samples affects load balance.
set -e
cd "$(dirname "$0")"
make -s all
REPS=${REPS:-3}
N=${N:-100000000}
PLIST=${PLIST:-"1 2 4 8"}
MPIRUN="mpirun --oversubscribe --bind-to none"
OUT=../../results/cs5_raw.txt
: > $OUT

$MPIRUN -np 2 ./pingpong > ../../results/cs5_pingpong.txt

for d in u z; do
  for r in $(seq $REPS); do ./serial $N $d >> $OUT; done
  for p in $PLIST; do for r in $(seq $REPS); do $MPIRUN -np $p ./parallel $N $d 1 >> $OUT; done; done
done

# oversampling vs. load balance (p = 4 and 8; a single run each is enough for counts)
for d in u z; do for p in 4 8; do for os in 1 4 16 64; do
  echo "OS $($MPIRUN -np $p ./parallel $N $d $os)" >> $OUT
done; done; done
echo "done -> $OUT"
