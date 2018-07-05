import sys
import types

__version__ = '0.1.0'

# define functions for pickling class methods
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    # Remove the pool object from the emcee sampler objects
    if obj.__class__.__name__=='EnsembleMCMC' or obj.__class__.__name__=='PTMCMC':
        obj.sampler.pool = None
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
else: # Attempt to make this work in py3 but it doesn't seem to do it
    import copyreg
    copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
