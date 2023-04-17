
# distutils: language=c++

from libcpp.algorithm cimport copy_n
from libcpp.vector cimport vector

from execution cimport par
from algorithm cimport sort


def cppsort(int[:] x):
    """
    Sort the elements of x "in-place" using std::sort
    """
    cdef vector[int] vec
    vec.resize(len(x))
    copy_n(&x[0], len(x), vec.begin())
    sort(par, vec.begin(), vec.end())
    copy_n(vec.begin(), len(x), &x[0])
