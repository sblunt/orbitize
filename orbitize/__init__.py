import sys
import types
import os

# try:
#     c = 1
#     frames = []
#     while sys._getframe(c).f_globals.get('__name__'):
#         print("stack: ", sys._getframe(c).f_globals.get('__name__'))
#         c+= 1
# except:
#     pass

# print ('I am being imported by', frames)

__version__ = '1.16.1'

# set Python env variable to keep track of example data dir
orbitize_dir = os.path.dirname(__file__)
DATADIR = os.path.join(orbitize_dir, 'example_data/')

# define functions for pickling class methods
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


if sys.version_info[0] < 3:
    import copy_reg
    copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    cuda_ext = True
except:
    cuda_ext = False