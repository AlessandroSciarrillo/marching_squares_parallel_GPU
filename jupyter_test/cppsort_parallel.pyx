
# distutils: language=c++
# distutils: libraries=tbb

from execution cimport par
from algorithm cimport sort

def cppsort(int[:] x):
    sort(par, &x[0], &x[-1] + 1)
