# Contributing

1. Fork or branch from `main`.
2. Build and check everything before opening a pull request:
   ```bash
   make && make check
   ```
3. Keep serial and parallel builds on the same optimisation flags, and keep the serial program as
   the reference: a parallel change must still pass its kernel's `make check`.
4. If a change affects performance, re-run the kernel's `run_experiments.sh` on an idle machine
   (median of 3 runs), note the machine in the pull request, and regenerate the figures with
   `python3 results/analyze.py`.
5. Comment the *why* (decomposition, ownership, communication) rather than restating the code.

CI builds all kernels and runs the correctness checks on every push and pull request.
