cuda-compile-4-cuda-and-noncuda
===============================

Design a CUDA-program able to run on a non-CUDA computer. When dealing with a non-CUDA computer it should use CPU version of the code to do the same operation.

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

This code is designed to build a Python extension. However if you want to create a standalone program the same technics apply:

1. tweak the Makefile to check for CUDA
2. if CUDA: use nvcc
3. if not CUDA: use g++ or gcc with option '-x c++ -c' in front of each *.cu file (for c++ code) / '-x c -c' (for c code)

Please refer to `src/Makefile` for further details.
