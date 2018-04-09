import numpy as np
cimport numpy as np

cdef extern from "kepler.hh": 
    void newton_array(const int start,
          const int end,
          const int threads, 
          const double manom[], 
          const double ecc[], 
          const double tol, 
          const int max_iter, 
          double eanom[])


def _cpp_newton_solver(np.ndarray[np.double_t,ndim=1] manom,
                    np.ndarray[np.double_t,ndim=1] ecc, 
                    tolerance = 1e-9, 
                    max_iter = 100, 
                    np.ndarray[np.double_t,ndim=1] eanom0 = None,
                    threads = 1):
    """
    Wrapper function for C++ implementation of Newton-Raphson solver for eccentric anomaly.
    Args:
        manom (np.array): array of mean anomalies
        ecc (np.array): array of eccentricities
        eanom0 (np.array): array of first guess for eccentric anomaly, same shape as manom (optional)
    Return:
        eanom (np.array): array of eccentric anomalies

    Written: Devin Cody, 2018

    """

    # Initialize at E=M, E=pi is better at very high eccentricities
    cdef np.ndarray eanom
    if eanom0 is None:
        eanom = np.copy(manom)
    else:
        eanom = np.copy(eanom0)


    newton_array(0, len(manom), threads, <double*> manom.data, <double*> ecc.data, tolerance, max_iter, <double*> eanom.data)

    return eanom