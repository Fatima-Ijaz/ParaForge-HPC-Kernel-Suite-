CXX      = g++
MPICXX   = mpicxx
MPICC    = mpicc
CXXFLAGS = -O3 -march=native -std=c++17 -Wall
CFLAGS   = -O3 -march=native -std=gnu11 -Wall
MPIRUN  ?= mpirun --oversubscribe

all: serial parallel pingpong

serial: serial.cpp keys.h
	$(CXX) $(CXXFLAGS) -o $@ serial.cpp

parallel: parallel.cpp keys.h
	$(MPICXX) $(CXXFLAGS) -o $@ parallel.cpp

pingpong: pingpong.c
	$(MPICC) $(CFLAGS) -o $@ pingpong.c

check: all
	./serial 10000000 u
	for np in 1 2 3 4; do for d in u z; do $(MPIRUN) -np $$np ./parallel 10000000 $$d 4; done; done

clean:
	rm -f serial parallel pingpong

.PHONY: all check clean
