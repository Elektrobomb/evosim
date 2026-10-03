import os

# One numba thread per physical core: SMT siblings cost more in sync than they add.
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))
