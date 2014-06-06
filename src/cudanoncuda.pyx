import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref

np.import_array()

cdef extern from "cudanoncuda-src.h":
    np.ndarray c_add(double*, double*, unsigned int)

def add(np.ndarray[double, ndim=1, mode="c"] v1 not None, np.ndarray[double, ndim=1, mode="c"] v2 not None):
    return c_add(<double*> v1.data, <double*> v2.data, v1.shape[0])

