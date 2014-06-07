import os
import sys
from os.path import join
from setuptools import setup, find_packages
from distutils.command.build_ext import build_ext
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# CUDA compilation is adapted from the source
# https://github.com/rmcgibbo/npcuda-example
# CUDA functions for compilation
def find_in_path(name, path):
    """
    Find a file in a search path
    """
    
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """
    Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
    
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))
    cudaconfig = {'home': home, 'nvcc': nvcc, 'include': join(home, 'include'), 'lib64': join(home, 'lib64')}
    
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

def customize_compiler_for_nvcc(self):
    """
    inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        # if nvcc is not available on this computer
        # temporarily change extension from .cu to .cpp (by creating a symlink to the original file)
        # a better way to do so, would be to change cc_args so that it ends by: -x c++ -c
        # unfortunately it does not work
        if not CUDA and ext == ".cu":
            original_src = src
            src = src[:-3] + ".cpp"
            ext = ".cpp"
            if os.path.lexists(src):
                os.remove(src)
            os.symlink(join(os.getcwd(), original_src), src)
        
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']
        
        try:
            super(obj, src, ext, cc_args, postargs, pp_opts)
        finally:        
            # remove the created symlink
            if not CUDA and os.path.islink(src):
                os.remove(src)
        
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class cuda_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

def find_files_with_ext(path, ext):
    files_list = list()
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                current_ext = name.split(".")[-1]
            except IndexError:
                continue
            
            if current_ext == ext:
                files_list.append(join(root, name))
    return files_list

# Locate CUDA paths
try:
    CUDA = locate_cuda()
except EnvironmentError, e:
    CUDA = None
    print("CUDA is not installed in your environment. The CPU version of this code will be compiled.")
    print(e)

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# ---- C/C++ EXTENSIONS ---- #

# Build extension
if CUDA:
    module_ext = Extension(name="src/cudanoncuda",
        sources=["src/cudanoncuda-src.cu", "src/cudanoncuda.pyx"],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={
            'gcc': [],
            'nvcc': ['-arch=sm_20', '--use_fast_math', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
        extra_link_args=['-lcudadevrt', '-lcudart'],
        include_dirs=[numpy_include, CUDA['include'], 'src'])
else:
    os.environ["CU"] = 'gcc'
    module_ext = Extension(name="src/cudanoncuda",
        sources=["src/cudanoncuda-src.cu", "src/cudanoncuda.pyx"],
        language='c++',
        extra_compile_args={'gcc': []},
        include_dirs=[numpy_include, 'src'])
    
cython_exts = [module_ext]

setup(name="compile4cudanoncuda",
    cmdclass={'build_ext': cuda_build_ext,},
    description='Design a CUDA-program able to run on a non-CUDA computer. When dealing with a non-CUDA computer it should use CPU version of the code to do the same operation.',
    author='Nicolas Dubien',
    author_email='github@dubien.org',
    include_dirs=[numpy_include],
    ext_modules=cythonize(cython_exts, quiet=True),
    packages=find_packages(),
    install_requires=[
        'Cython>=0.20.1',
        'numpy>=1.8.0',
    ],
    tests_require=[], 
    zip_safe=False,
)
