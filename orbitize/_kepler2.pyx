import numpy as np
cimport numpy as np

np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef extern from "kepler.c": 
    void newton_array(const int n_elements,
                        const double manom[], 
                        const double ecc[], 
                        const double tol, 
                        const int max_iter, 
                        double eanom[])

cdef extern from "kepler.c":
    void mikkola_array(const int n_elements,
                        const double manom[],
                        const double ecc[],
                        double eanom[])

cdef _c_newton_solver(np.ndarray[DTYPE_t,ndim=1] manom,
                    np.ndarray[DTYPE_t,ndim=1] ecc, 
                    float tolerance = 1e-9, 
                    int max_iter = 100, 
                    # np.ndarray[DTYPE_t,ndim=1] eanom0 = None
                    ):
    """
    Wrapper function for C implementation of Newton-Raphson solver for eccentric anomaly.
    Args:
        manom (np.array): array of mean anomalies
        ecc (np.array): array of eccentricities
        eanom0 (np.array): array of first guess for eccentric anomaly, same shape as manom (optional)
    Return:
        eanom (np.array): array of eccentric anomalies
    Written: Devin Cody, 2018
    """

    # Initialize at E=M, E=pi is better at very high eccentricities
    cdef np.ndarray[DTYPE_t, ndim=1] eanom
    # if eanom0 is None:
    eanom = np.copy(manom)
    # else:
    #     eanom = np.copy(eanom0)


    newton_array(manom.shape[0], <double*> manom.data, <double*> ecc.data, tolerance, max_iter, <double*> eanom.data)

    return eanom

cdef _c_mikkola_solver(np.ndarray[DTYPE_t,ndim=1] manom,
                      np.ndarray[DTYPE_t,ndim=1] ecc):
    """
    Wrapper function for C implementation of Newton-Raphson solver for eccentric anomaly.
    Args:
        manom (np.array): array of mean anomalies
        ecc (np.array): array of eccentricities
        eanom0 (np.array): array of first guess for eccentric anomaly, same shape as manom (optional)
    Return:
        eanom (np.array): array of eccentric anomalies
    Written: Devin Cody, 2018
    """

    # Initialize at E=M, E=pi is better at very high eccentricities
    cdef np.ndarray[DTYPE_t, ndim=1] eanom
    eanom = np.zeros(manom.shape[0])

    mikkola_array(manom.shape[0], <double*> manom.data, <double*> ecc.data, <double*> eanom.data)

    return eanom

"""
This module solves for the orbit of the planet given Keplerian parameters.
"""
import astropy.units as u
import astropy.constants as consts

PERIOD = np.sqrt(
        4 * np.pi**2.0 * (1.0 * u.AU)**3 /
        (consts.G * (1.0 * u.Msun))
    )
cdef double PERIOD_CONVERSION_FACTOR = PERIOD.value / PERIOD.to(u.day).value
cdef double G = consts.G.value

KV = np.sqrt(consts.G) * (1.0 * u.Msun) / np.sqrt(1.0 * u.Msun) / np.sqrt(1.0 * u.au)
cdef double KV_CONVERSION_FACTOR = KV.value / KV.to(u.km/u.s).value


def tau_to_manom(
    np.ndarray[DTYPE_t, ndim=2] date,
    np.ndarray[DTYPE_t, ndim=1] sma,
    np.ndarray[DTYPE_t, ndim=1] mtot,
    np.ndarray[DTYPE_t, ndim=1] tau,
    float tau_ref_epoch):
    """
    Gets the mean anomlay
    
    Args:
        date (float or np.array): MJD
        sma (float): semi major axis (AU)
        mtot (float): total mass (M_sun)
        tau (float): epoch of periastron, in units of the orbital period
        tau_ref_epoch (float): reference epoch for tau
        
    Returns:
        float or np.array: mean anomaly on that date [0, 2pi)
    """

    cdef np.ndarray[DTYPE_t, ndim=1] period = np.sqrt(
        4 * np.pi**2.0 * (sma**3 )/
        (G * mtot)
    )
    period = period / PERIOD_CONVERSION_FACTOR

    cdef np.ndarray[DTYPE_t, ndim=2] frac_date = (date - tau_ref_epoch)/period
    frac_date %= 1

    cdef np.ndarray[DTYPE_t, ndim=2] mean_anom = (frac_date - tau) * 2 * np.pi
    mean_anom %= 2 * np.pi

    return mean_anom


def calc_orbit(
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

    """
    Returns the separation and radial velocity of the body given array of
    orbital parameters (size n_orbs) at given epochs (array of size n_dates)

    Based on orbit solvers from James Graham and Rob De Rosa. Adapted by Jason Wang and Henry Ngo.

    Args:
        epochs (np.array): MJD times for which we want the positions of the planet
        sma (np.array): semi-major axis of orbit [au]
        ecc (np.array): eccentricity of the orbit [0,1]
        inc (np.array): inclination [radians]
        aop (np.array): argument of periastron [radians]
        pan (np.array): longitude of the ascending node [radians]
        tau (np.array): epoch of periastron passage in fraction of orbital period past MJD=0 [0,1]
        plx (np.array): parallax [mas]
        mtot (np.array): total mass of the two-body orbit (M_* + M_planet) [Solar masses]
        mass_for_Kamp (np.array, optional): mass of the body that causes the RV signal.
            For example, if you want to return the stellar RV, this is the planet mass.
            If you want to return the planetary RV, this is the stellar mass. [Solar masses].
            For planet mass ~ 0, mass_for_Kamp ~ M_tot, and function returns planetary RV (default).
        tau_ref_epoch (float, optional): reference date that tau is defined with respect to (i.e., tau=0)
        tolerance (float, optional): absolute tolerance of iterative computation. Defaults to 1e-9.
        max_iter (int, optional): maximum number of iterations before switching. Defaults to 100.
        use_c (bool, optional): Use the C solver if configured. Defaults to True
        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False

    Return:
        3-tuple:

            raoff (np.array): array-like (n_dates x n_orbs) of RA offsets between the bodies
            (origin is at the other body) [mas]

            deoff (np.array): array-like (n_dates x n_orbs) of Dec offsets between the bodies [mas]

            vz (np.array): array-like (n_dates x n_orbs) of radial velocity of one of the bodies
                (see `mass_for_Kamp` description)  [km/s]

    Written: Jason Wang, Henry Ngo, 2018
    """
    # n_orbs = np.size(sma)  # num sets of input orbital parameters
    cdef int n_dates = epochs.shape[0]  # number of dates to compute offsets and vz

    # return planetary RV if `mass_for_Kamp` is not defined
    # if mass_for_Kamp is None:
    #     mass_for_Kamp = mtot

    # Necessary for _calc_ecc_anom, for now
    # if np.isscalar(epochs):  # just in case epochs is given as a scalar
    #     epochs = np.array([epochs])
    cdef np.ndarray[DTYPE_t,ndim=2] ecc_arr = np.tile(ecc, (n_dates, 1))

    # # compute mean anomaly (size: n_orbs x n_dates)
    cdef np.ndarray[DTYPE_t,ndim=2] manom = tau_to_manom(epochs[:, None], sma, mtot, tau, tau_ref_epoch)
    # compute eccentric anomalies (size: n_orbs x n_dates)
    cdef np.ndarray[DTYPE_t,ndim=2] eanom = _calc_ecc_anom(manom, ecc_arr, tolerance=tolerance, max_iter=max_iter)

    # compute the true anomalies (size: n_orbs x n_dates)
    # Note: matrix multiplication makes the shapes work out here and below
    cdef np.ndarray[DTYPE_t,ndim=2] tanom = 2.*np.arctan(np.sqrt((1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eanom))
    # compute 3-D orbital radius of second body (size: n_orbs x n_dates)
    cdef np.ndarray[DTYPE_t,ndim=2] radius = sma * (1.0 - ecc * np.cos(eanom))

    # compute ra/dec offsets (size: n_orbs x n_dates)
    # math from James Graham. Lots of trig
    cdef np.ndarray[DTYPE_t,ndim=1] c2i2 = np.cos(0.5*inc)**2
    cdef np.ndarray[DTYPE_t,ndim=1] s2i2 = np.sin(0.5*inc)**2
    cdef np.ndarray[DTYPE_t,ndim=2] arg1 = tanom + aop + pan
    cdef np.ndarray[DTYPE_t,ndim=2] arg2 = tanom + aop - pan
    cdef np.ndarray[DTYPE_t,ndim=2] c1 = np.cos(arg1)
    cdef np.ndarray[DTYPE_t,ndim=2] c2 = np.cos(arg2)
    cdef np.ndarray[DTYPE_t,ndim=2] s1 = np.sin(arg1)
    cdef np.ndarray[DTYPE_t,ndim=2] s2 = np.sin(arg2)

    # updated sign convention for Green Eq. 19.4-19.7
    cdef np.ndarray[DTYPE_t,ndim=2] raoff = radius * (c2i2*s1 - s2i2*s2) * plx
    cdef np.ndarray[DTYPE_t,ndim=2] deoff = radius * (c2i2*c1 + s2i2*c2) * plx

    # compute the radial velocity (vz) of the body (size: n_orbs x n_dates)
    # first comptue the RV semi-amplitude (size: n_orbs x n_dates)
    cdef np.ndarray[DTYPE_t,ndim=1] Kv = np.sqrt(G / (1.0 - ecc**2)) * (mass_for_Kamp *
                                               np.sin(inc)) / np.sqrt(mtot) / np.sqrt(sma)
    # Convert to km/s
    Kv = Kv / KV_CONVERSION_FACTOR

    # compute the vz
    cdef np.ndarray[DTYPE_t,ndim=2] vz = Kv * (ecc*np.cos(aop) + np.cos(aop + tanom))
    return raoff, deoff, vz

cdef _calc_ecc_anom(manom, ecc, tolerance=1e-9, max_iter=100):
    """
    Computes the eccentric anomaly from the mean anomlay.
    Code from Rob De Rosa's orbit solver (e < 0.95 use Newton, e >= 0.95 use Mikkola)

    Args:
        manom (float/np.array): mean anomaly, either a scalar or np.array of any shape
        ecc (float/np.array): eccentricity, either a scalar or np.array of the same shape as manom
        tolerance (float, optional): absolute tolerance of iterative computation. Defaults to 1e-9.
        max_iter (int, optional): maximum number of iterations before switching. Defaults to 100.
        use_c (bool, optional): Use the C solver if configured. Defaults to False
        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False

Return:
        eanom (float/np.array): eccentric anomalies, same shape as manom

    Written: Jason Wang, 2018
    """

    if np.isscalar(ecc) or (np.shape(manom) == np.shape(ecc)):
        pass
    else:
        raise ValueError("ecc must be a scalar, or ecc.shape == manom.shape")

    # If manom is a scalar, make it into a one-element array
    if np.isscalar(manom):
        manom = np.array((manom, ))

    # If ecc is a scalar, make it the same shape as manom
    if np.isscalar(ecc):
        ecc = np.full(np.shape(manom), ecc)

    # Initialize eanom array
    eanom = np.full(np.shape(manom), np.nan)

    # Save some boolean arrays
    ecc_zero = ecc == 0.0
    ecc_low = ecc < 0.95

    # First deal with e == 0 elements
    ind_zero = np.where(ecc_zero)
    if len(ind_zero[0]) > 0:
        eanom[ind_zero] = manom[ind_zero]

    # Now low eccentricities
    ind_low = np.where(~ecc_zero & ecc_low)
    if len(ind_low[0]) > 0: 
        eanom[ind_low] = _newton_solver_wrapper(manom[ind_low], ecc[ind_low], tolerance, max_iter)
    
    # Now high eccentricities
    ind_high = np.where(~ecc_zero & ~ecc_low | (eanom == -1)) # The C and CUDA solvers return the unphysical value -1 if they fail to converge
    if len(ind_high[0]) > 0: 
        eanom[ind_high] = _mikkola_solver_wrapper(manom[ind_high], ecc[ind_high])

    return eanom

def _newton_solver_wrapper(manom, ecc, tolerance, max_iter):
    """
    Wrapper for the various (Python, C, CUDA) implementations of the Newton-Raphson solver 
    for eccentric anomaly.

    Args:
        manom (np.array): array of mean anomalies
        ecc (np.array): array of eccentricities
        eanom0 (np.array, optional): array of first guess for eccentric anomaly, same shape as manom (optional)
        use_c (bool, optional): Use the C solver if configured. Defaults to False
        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False
    Return:
        eanom (np.array): array of eccentric anomalies

    Written: Devin Cody, 2021
    """
    eanom = np.empty_like(manom)
    
    eanom = _c_newton_solver(manom, ecc, tolerance=tolerance, max_iter=max_iter)

    return eanom

def _mikkola_solver_wrapper(manom, ecc):
    """
    Wrapper for the various (Python, C, CUDA) implementations of Analtyical Mikkola solver 

    Args:
        manom (np.array): array of mean anomalies between 0 and 2pi
        ecc (np.array): eccentricity
        use_c (bool, optional): Use the C solver if configured. Defaults to False
        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False


    Return:
        eanom (np.array): array of eccentric anomalies

    Written: Jason Wang, 2018
    """

    ind_change = np.where(manom > np.pi)
    manom[ind_change] = (2.0 * np.pi) - manom[ind_change]
    eanom = _c_mikkola_solver(manom, ecc)
    eanom[ind_change] = (2.0 * np.pi) - eanom[ind_change]

    return eanom
