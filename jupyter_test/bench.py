import timeit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import format_time

plt.rcParams.update({'font.size': 15})

def random_data(size):
    return np.random.randint(size, size=size, dtype="int32")

def time_sort(size, sort_func):
    repeat = 10
    dt = 0
    for i in range(repeat):
        data = random_data(size)
        start = timeit.default_timer()
        sort_func(data)
        end = timeit.default_timer()
        dt += (end - start)
    return (dt) / repeat

def run_benchmark(sort_func, name):
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    return pd.Series(
        [time_sort(s, sort_func) for s in sizes],
        index=sizes,
        name=name
    )

# 1

from cppsort_serial import cppsort

cpu_serial = run_benchmark(cppsort, "CPU serial")
cpu_serial.map(format_time).to_frame()

# 2

try:
    from cppsort_parallel import cppsort

    cpu_parallel = run_benchmark(cppsort, "CPU parallel")
    cpu_parallel.map(format_time).to_frame()
except ImportError:  # handle if TBB is missing
    cpu_parallel = cpu_serial.replace(cpu_serial.values, pd.NA).rename("CPU parallel")

# 3

#from cppsort_stdpar import cppsort

gpu_stdpar = run_benchmark(cppsort, "GPU (nvc++ with -stdpar)")
gpu_stdpar.map(format_time).to_frame()


baseline = run_benchmark(lambda x: x.sort(), "NumPy")

all_timings = pd.DataFrame([baseline, cpu_serial, cpu_parallel, gpu_stdpar]).T
speedups = (1/all_timings).multiply(baseline, axis=0)

all_timings.applymap(format_time)

speedups.iloc[:, 1:].plot(
    kind="bar", 
    xlabel="Number of elements in input array", 
    ylabel="Speedup v/s NumPy .sort()", 
    figsize=(10, 5),
    grid=True
)
plt.xticks(rotation=0)
plt.savefig("sort.png")