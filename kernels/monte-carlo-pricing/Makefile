CC      = gcc
MPICC   = mpicc
CFLAGS  = -O3 -march=native -std=gnu11 -Wall
LDLIBS  = -lm

all: serial parallel parallel_mpi

serial: serial.c mc_common.h
	$(CC) $(CFLAGS) -o $@ serial.c $(LDLIBS)

parallel: parallel.c mc_common.h
	$(CC) $(CFLAGS) -fopenmp -o $@ parallel.c $(LDLIBS)

parallel_mpi: parallel_mpi.c mc_common.h
	$(MPICC) $(CFLAGS) -o $@ parallel_mpi.c $(LDLIBS)

# serial and 1-thread parallel must print the same price; p=2 twice must agree with itself
check: all
	./serial 10000000
	OMP_NUM_THREADS=1 ./parallel 10000000
	OMP_NUM_THREADS=2 ./parallel 10000000
	OMP_NUM_THREADS=2 ./parallel 10000000
	mpirun --oversubscribe -np 2 ./parallel_mpi 10000000

clean:
	rm -f serial parallel parallel_mpi

.PHONY: all check clean
