import os

__version__ = '2.1.0'

# set Python env variable to keep track of example data dir
orbitize_dir = os.path.dirname(__file__)
DATADIR = os.path.join(orbitize_dir, 'example_data/')

# Detect a valid CUDA environment
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    cuda_ext = True
except:
    cuda_ext = False

try:
    from . import _kepler
    cext = True
except ImportError:
    cext = False
