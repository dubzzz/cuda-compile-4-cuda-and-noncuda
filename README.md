#Compile 4 CUDA & Non-CUDA

Design a CUDA-program able to run on a non-CUDA computer. When dealing with a non-CUDA computer it should use CPU version of the code to do the same operation.

##How does it work?

This code is designed to build a Python extension. However if you want to create a standalone program the same technics apply:

1. tweak the Makefile to check for CUDA
2. if CUDA: use nvcc
3. if not CUDA: use g++ or gcc with the option '-x c++ -c' in front of each *.cu file (for c++ code) / '-x c -c' (for c code)

Please refer to `src/Makefile` for further details.

Please note that you also need to adapt your C/C++ sourcecode in order to be able to cope with cases where CUDA does not exist. In order to do so I used the `__CUDACC__` variable which is defined if and only if using nvcc.

When compiling with another compiler (different from nvcc), every reference to CUDA code need to be removed. Please refer to `src/cudanoncuda-src.cu`.

##Special cases

Even if nvcc exists in the system, it is not sure that it can be used for GPU computations.

Compiling with `nvcc` or having a copy of `libcudart.so` does not prove that the program will be able to launch a kernel on your system. Moreover launching a CUDA program and having `libcudart.so` on the computer will not raise any exceptions when launching kernels, except cuda errors which need to be handle manually. For that reason, the C/C++ code needs to check GPU cards at runtime. 

##Test it!

Here is a sample code to test the behaviour of the generated library:
```Python
import numpy as np
import cudanoncuda as cnc

# Create vectors
va = np.random.random(5)
vb = np.random.random(5)

# Use the library
# Depending on your environment it should run CUDA or CPU code
vc = cnc.add(va, vb)
```

Remark: a code compiled on a computer with `nvcc` on it and executed on another computer without CUDA (or more precisely: without `libcudart.so`) will fail to run.
