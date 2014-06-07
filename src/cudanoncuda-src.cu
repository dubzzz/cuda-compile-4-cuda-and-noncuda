/**
    The purpose of this algorithm is not to design a clean CUDA-code which performs
    checks on input data and cuda error recovery.
    
    The idea is just to design a simple way of dealing with CUDA and non-CUDA computers using a single sourcecode
    
    __CUDACC__: defines whether nvcc is steering compilation or not
    __CUDA_ARCH__: is always undefined when compiling host code, steered by nvcc or not
    __CUDA_ARCH__: is only defined for the device code trajectory of compilation steered by nvcc
    //for further details: http://stackoverflow.com/questions/8796369/cuda-and-nvcc-using-the-preprocessor-to-choose-between-float-or-double
*/

#include "cudanoncuda-src.h"
#include <stdio.h>

void init_numpy();
PyArrayObject *c_add(const double* v1, const double* v2, const unsigned int &size);
void c_add_cpu(const double* v1, const double* v2, const unsigned int &size, double* vres);

#ifdef __CUDACC__
    #define CUDA_MAJOR 2
    #define CUDA_MINOR 0
    int __DEVICE_COUNT__(-1); // -1 means to be done
    bool is_cuda_available();
    void c_add_cuda(const double* h_v1, const double* h_v2, const unsigned int &size, double* h_vres);
    __global__ void c_add_cuda_kernel(const double* d_v1, const double* d_v2, const unsigned int size, double* d_vres);
#endif

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
    printf("PyArrayObject *c_add(const double*, const double*, const unsigned int&)\n");
    init_numpy(); // init NumPy if not already done
    
    // Define NumPy array
    int dims[] = {size};
    PyArrayObject *vres = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_DOUBLE);
    
    #ifdef __CUDACC__
        if (is_cuda_available())
        {
            // Apply CUDA version of c_add
            c_add_cuda(v1, v2, size, (double*)vres->data);
        }
        else
        {
            // Apply CPU version of c_add
            c_add_cpu(v1, v2, size, (double*)vres->data);
        }
    #else
        // Apply CPU version of c_add
        c_add_cpu(v1, v2, size, (double*)vres->data);
    #endif
    
    return vres;
}

#ifdef __CUDACC__

/**
    Check if CUDA is available or not
    and compatible or not
    
    The check itself is only done one time.
    Result is then stored in the variable __DEVICE_COUNT__
*/
bool is_cuda_available()
{
    printf("  bool is_cuda_available()\n");
    if (__DEVICE_COUNT__ == -1)
    {
        printf("    performs the check\n");
        int deviceCount;
        cudaError_t e = cudaGetDeviceCount(&deviceCount);
        if (e != cudaSuccess)
            __DEVICE_COUNT__ = 0;
        else
        {
            printf("    %d GPU found:\n", deviceCount);
            __DEVICE_COUNT__ = 0;
            
            // for each GPU check if it has the required compute capability
            for (int i(0) ; i != deviceCount ; i++)
            {
                cudaDeviceProp prop;
                e = cudaGetDeviceProperties(&prop, i);
                if (e != cudaSuccess)
                    printf("      fails to access device #%d\n", i);
                else
                {
                    printf("      found: %s, with CUDA Compute Capability: %d.%d\n", prop.name, prop.major, prop.minor);
                    // check compute capability
                    if (prop.major > CUDA_MAJOR || (prop.major == CUDA_MAJOR && prop.minor >= CUDA_MINOR))
                    {
                        printf("        has the required compute capability\n");
                        cudaSetDevice(i);
                        __DEVICE_COUNT__++;
                    }
                }
            }
        }
    }
    return __DEVICE_COUNT__ != 0;
}

void c_add_cuda(const double* h_v1, const double* h_v2, const unsigned int &size, double* h_vres)
{
    printf("  void c_add_cuda(const double*, const double*, const unsigned int&, double*)\n");
    
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

#endif

void c_add_cpu(const double* v1, const double* v2, const unsigned int &size, double* vres)
{
    printf("  void c_add_cpu(const double*, const double*, const unsigned int&, double*)\n");
    
    for (unsigned int i(0) ; i != size ; i++)
        vres[i] = v1[i] + v2[i];
}

