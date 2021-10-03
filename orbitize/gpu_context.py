import os
import sys
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class gpu_context:
    """
    GPU context which manages the allocation of memory, the movement of memory between python and the GPU, 
    and the calling of GPU funcitons

    Written: Devin Cody, 2021
    """
    def __init__(self, len_gpu_arrays = 10000000):
        self.gpu_initalized = False
        self.len_gpu_arrays = len_gpu_arrays

        try:
            print("Compiling kernel")
            if "win" in sys.platform:
                f_newton = open(os.path.dirname(__file__) + "\\kernels\\newton.cu", 'r')
                f_mikkola = open(os.path.dirname(__file__) + "\\kernels\\mikkola.cu", 'r')
            else:
                f_newton = open(os.path.dirname(__file__) + "/kernels/newton.cu", 'r')
                f_mikkola = open(os.path.dirname(__file__) + "/kernels/mikkola.cu", 'r')

            fstr_newton = "".join(f_newton.readlines())
            mod_newton = SourceModule(fstr_newton)
            self.newton_gpu = mod_newton.get_function("newton_gpu")

            fstr_mikkola = "".join(f_mikkola.readlines())
            mod_mikkola = SourceModule(fstr_mikkola)
            self.mikkola_gpu = mod_mikkola.get_function("mikkola_gpu")

            print("Allocating with {} bytes".format(self.len_gpu_arrays))
            self.tolerance = np.array([1e-9], dtype = np.float64)
            self.max_iter = np.array([100])
            self.eanom = None

            self.d_manom = cuda.mem_alloc(self.len_gpu_arrays)
            self.d_ecc = cuda.mem_alloc(self.len_gpu_arrays)
            self.d_eanom = cuda.mem_alloc(self.len_gpu_arrays)

            self.d_tol = cuda.mem_alloc(self.tolerance.nbytes)
            self.d_max_iter = cuda.mem_alloc(self.max_iter.nbytes)
            
            print("Copying parameters to GPU")
            cuda.memcpy_htod(self.d_tol, self.tolerance)
            cuda.memcpy_htod(self.d_max_iter, self.max_iter)
            gpu_initalized = True
        except Exception as e:
            print("Error: KEPLER: Unable to initialize Kepler GPU solver context")
            raise(e)

    def newton(self, manom, ecc, eanom, eanom0 = None, tolerance=1e-9, max_iter=100):
        """
        Moves numpy arrays onto the GPU memory, calls the Newton-Raphson solver for eccentric anomaly
        and copies the result back into a numpy array.

        Args:
            manom (np.array): array of mean anomalies
            ecc (np.array): array of eccentricities
            eanom (np.array): array of eccentric anomalies (return by reference)
            eanom0 (np.array, optional): array of first guess for eccentric anomaly, same shape as manom (optional)
        Return:
            None: eanom is changed by reference

        Written: Devin Cody, 2021

        """
        # Check to make sure we have enough data to process orbits
        if (self.len_gpu_arrays < manom.nbytes):
            self.len_gpu_arrays = manom.nbytes
            self.d_manom = cuda.mem_alloc(self.len_gpu_arrays)
            self.d_ecc = cuda.mem_alloc(self.len_gpu_arrays)
            self.d_eanom = cuda.mem_alloc(self.len_gpu_arrays)

        cuda.memcpy_htod(self.d_manom, manom)
        cuda.memcpy_htod(self.d_ecc, ecc)
        cuda.memcpy_htod(self.d_tol, tolerance)
        cuda.memcpy_htod(self.d_max_iter, max_iter)

        # Initialize at E=M, E=pi is better at very high eccentricities
        if eanom0 is None:
            cuda.memcpy_dtod(self.d_eanom, self.d_manom, self.len_gpu_arrays)
        else:
            cuda.memcpy_htod(self.d_eanom, eanom0)

        self.newton_gpu(self.d_manom, self.d_ecc, self.d_eanom, self.d_max_iter, self.d_tol, grid = (len(manom)//64+1,1,1), block = (64,1,1))
        cuda.memcpy_dtoh(eanom, self.d_eanom)

    def mikkola(self, manom, ecc, eanom):
        """
        Moves numpy arrays onto the GPU memory, calls the analtyical Mikkola solver for eccentric anomaly
        and copies the result back into a numpy array.
        
        Args:
            manom (np.array): array of mean anomalies between 0 and 2pi
            ecc (np.array): eccentricity
            eanom (np.array): array of eccentric anomalies (return by reference)
        Return:
            None: eanom is changed by reference

        Written: Devin Cody, 2021
        """
        # Check to make sure we have enough data to process orbits
        if (self.len_gpu_arrays < manom.nbytes):
            self.len_gpu_arrays = manom.nbytes
            self.d_manom = cuda.mem_alloc(self.len_gpu_arrays)
            self.d_ecc = cuda.mem_alloc(self.len_gpu_arrays)

        cuda.memcpy_htod(self.d_manom, manom)
        cuda.memcpy_htod(self.d_ecc, ecc)

        self.mikkola_gpu(self.d_manom, self.d_ecc, self.d_eanom, grid = (len(manom)//64+1,1,1), block = (64,1,1))
        cuda.memcpy_dtoh(eanom, self.d_eanom)