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
