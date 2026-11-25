# Serial code is built with plain gcc, the MPI code with the mpicc wrapper; both use the same
# optimisation flags so that speedups compare like with like.
CC      = gcc
MPICC   = mpicc
CFLAGS  = -O3 -march=native -std=gnu11 -Wall
LDLIBS  = -lm
NP     ?= 2
MPIRUN ?= mpirun --oversubscribe

all: serial parallel verify

serial: serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

parallel: parallel.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

verify: verify.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# Correctness check: run the serial solver once, then every decomposition / halo mode on
# 1, 2 and 4 ranks, and require the final grids to be identical (see verify.c).
check: all
	./serial 512 200 50 ref.bin
	for d in 1 2; do for m in b nb; do for np in 1 2 4; do \
	  $(MPIRUN) -np $$np ./parallel 512 200 50 $$d $$m out.bin > /dev/null && \
	  printf "decomp=$$d mode=$$m np=$$np : " && ./verify ref.bin out.bin 512 || exit 1; \
	done; done; done
	rm -f ref.bin out.bin

clean:
	rm -f serial parallel verify *.bin

.PHONY: all check clean
