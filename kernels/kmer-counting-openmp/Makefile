CC      = gcc
CFLAGS  = -O3 -march=native -std=gnu11 -Wall -Wno-unknown-pragmas
OMPFLAG = -fopenmp

all: serial parallel

serial: serial.c kmer_common.h
	$(CC) $(CFLAGS) -o $@ serial.c

parallel: parallel.c kmer_common.h
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ parallel.c

# quick correctness check (B, C, R must PASS; A is expected to FAIL with p>1)
check: all
	./serial 20000000 8
	for v in A B C R; do OMP_NUM_THREADS=2 ./parallel 20000000 8 u $$v; done
	OMP_NUM_THREADS=2 ./parallel 20000000 12 u C

clean:
	rm -f serial parallel *.bin

.PHONY: all check clean
