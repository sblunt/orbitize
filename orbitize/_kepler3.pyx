import numpy as np
cimport numpy as np

np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "kepler3.c": 
    void calc_orbit(
        const int n_orbits,
        const int n_epochs,
        const double epochs[],
        const double sma[],
        const double ecc[],
        const double inc[],
        const double aop[],
        const double pan[],
        const double tau[],
        const double plx[],
        const double mtot[],
        const double mass_for_Kamp[],
        const double tau_ref_epoch,
        const double tolerance,
        const int max_iter,
        double raoff[],
        double deoff[],
        double vz[]
    )

def _calc_orbit(
    np.ndarray[DTYPE_t,ndim=1] epochs,
    np.ndarray[DTYPE_t,ndim=1] sma,
    np.ndarray[DTYPE_t,ndim=1] ecc,
    np.ndarray[DTYPE_t,ndim=1] inc,
    np.ndarray[DTYPE_t,ndim=1] aop,
    np.ndarray[DTYPE_t,ndim=1] pan,
    np.ndarray[DTYPE_t,ndim=1] tau,
    np.ndarray[DTYPE_t,ndim=1] plx,
    np.ndarray[DTYPE_t,ndim=1] mtot,
    np.ndarray[DTYPE_t,ndim=1] mass_for_Kamp,
    float tau_ref_epoch=58849,
    float tolerance=1e-9, 
    int max_iter=100,
):
    cdef int n_orbits = sma.shape[0]
    cdef int n_epochs = epochs.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] raoff = np.zeros(n_orbits * n_epochs)
    cdef np.ndarray[DTYPE_t, ndim=1] deoff = np.zeros(n_orbits * n_epochs)
    cdef np.ndarray[DTYPE_t, ndim=1] vz = np.zeros(n_orbits * n_epochs)
    
    calc_orbit(
        n_orbits,
        n_epochs,
        <double*> epochs.data,
        <double*> sma.data,
        <double*> ecc.data,
        <double*> inc.data,
        <double*> aop.data,
        <double*> pan.data,
        <double*> tau.data,
        <double*> plx.data,
        <double*> mtot.data,
        <double*> mass_for_Kamp.data,
        tau_ref_epoch,
        tolerance,
        max_iter,
        <double*> raoff.data,
        <double*> deoff.data,
        <double*> vz.data)
    cdef np.ndarray[DTYPE_t, ndim=2] raoff_ret = raoff.reshape((n_orbits, n_epochs))
    cdef np.ndarray[DTYPE_t, ndim=2] deoff_ret = deoff.reshape((n_orbits, n_epochs))
    cdef np.ndarray[DTYPE_t, ndim=2] vz_ret = vz.reshape((n_orbits, n_epochs))
    return raoff_ret, deoff_ret, vz_ret