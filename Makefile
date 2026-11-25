# Top-level convenience Makefile: builds, checks or cleans every kernel in one go.
# Each kernel keeps its own Makefile, so `make -C kernels/<name>` works as well.

KERNELS := heat-diffusion-mpi kmer-counting-openmp monte-carlo-pricing pagerank-openmp sample-sort-mpi

.PHONY: all check clean $(KERNELS)

all: $(KERNELS)

$(KERNELS):
	$(MAKE) -C kernels/$@

# runs every kernel's correctness check against its serial baseline
check: all
	@for k in $(KERNELS); do \
	  echo "==== $$k"; \
	  $(MAKE) -C kernels/$$k check || exit 1; \
	done

clean:
	@for k in $(KERNELS); do $(MAKE) -C kernels/$$k clean; done
