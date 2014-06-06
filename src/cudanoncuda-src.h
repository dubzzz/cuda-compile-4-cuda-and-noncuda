#ifndef __CUDANONCUDA_SRC_H__
#define __CUDANONCUDA_SRC_H__

#include <Python.h>
#include <arrayobject.h>
#define MAX_THREADS 256

PyArrayObject *c_add(const double* v1, const double* v2, const unsigned int &size);

#endif
