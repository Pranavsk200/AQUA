import os

# small matrices: multi-threaded BLAS spins more than it helps (measured ~5x slower here)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
