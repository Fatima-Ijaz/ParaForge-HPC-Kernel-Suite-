CC      = gcc
CFLAGS  = -O3 -march=native -std=gnu11 -Wall
LDLIBS  = -lm
SCALE  ?= 20
EF     ?= 16

all: serial parallel

serial: serial.c graph.h
	$(CC) $(CFLAGS) -o $@ serial.c $(LDLIBS)

parallel: parallel.c graph.h
	$(CC) $(CFLAGS) -fopenmp -o $@ parallel.c $(LDLIBS)

# small-graph check: pull and push must match the serial ranks (L1 diff < 1e-9)
check: all
	./serial 16 16 ref16.bin
	OMP_NUM_THREADS=2 ./parallel 16 16 pull static 0 ref16.bin
	OMP_NUM_THREADS=2 ./parallel 16 16 push dynamic 64 ref16.bin
	OMP_NUM_THREADS=2 ./parallel 16 16 pull guided 0 ref16.bin
	rm -f ref16.bin

clean:
	rm -f serial parallel *.bin

.PHONY: all check clean
