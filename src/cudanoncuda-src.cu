/**
    The purpose of this algorithm is not to design a clean CUDA-code which performs
    checks on input data and cuda error recovery.
    
    The idea is just to design a simple way of dealing with CUDA and non-CUDA computers using a single sourcecode
*/

#include "cudanoncuda-src.h"
#include <stdio.h>

void init_numpy();
PyArrayObject *c_add(const double* v1, const double* v2, const unsigned int &size);
void c_add_cuda(const double* h_v1, const double* h_v2, const unsigned int &size, double* h_vres);
__global__ void c_add_cuda_kernel(const double* d_v1, const double* d_v2, const unsigned int size, double* d_vres);

/**
    import_array has to be call
    before the first call to NumPy API
*/
bool is_init(false);
void init_numpy()
{
    if (! is_init)
    {
        import_array();
        is_init = true;
    }
}

PyArrayObject *c_add(const double* v1, const double* v2, const unsigned int &size)
{
    printf("PyArrayObject *c_add(double*, double*, const unsigned int&)\n");
    init_numpy(); // init NumPy if not already done
    
    // Define NumPy array
    int dims[] = {size};
    PyArrayObject *vres = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_DOUBLE);
    
    // Apply CUDA version of c_add
    c_add_cuda(v1, v2, size, (double*)vres->data);
    
    return vres;
}

void c_add_cuda(const double* h_v1, const double* h_v2, const unsigned int &size, double* h_vres)
{
    // Build CUDA copies of the host arrays
    double *d_v1, *d_v2, *d_vres;
    cudaMalloc(&d_v1, size * sizeof(double)); // malloc + memcpy
    cudaMemcpy(d_v1, h_v1, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_v2, size * sizeof(double)); // malloc + memcpy
    cudaMemcpy(d_v2, h_v2, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_vres, size * sizeof(double)); // only malloc
    
    // Run CUDA kernel
    c_add_cuda_kernel<<<(size + MAX_THREADS -1)/MAX_THREADS, MAX_THREADS>>>(d_v1, d_v2, size, d_vres);
    cudaThreadSynchronize(); // block until the device is finished
    
    // Copy result to CPU
    cudaMemcpy(h_vres, d_vres, size * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free CUDA copies
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_vres);
}

__global__ void c_add_cuda_kernel(const double* d_v1, const double* d_v2, const unsigned int size, double* d_vres)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    d_vres[i] = d_v1[i] + d_v2[i];    
}

