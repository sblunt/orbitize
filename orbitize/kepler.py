"""
This module solves for the orbit of the planet given Keplerian parameters
"""
import numpy as np
import astropy.units as u
import astropy.constants as consts


def calc_orbit(epochs, sma, ecc, tau, argp, lan, inc, plx, mtot, mass=None):
    """
    Returns the separation and radial velocity of the body given array of
    orbital parameters (size n_orbs) at given epochs (array of size n_dates)

    Based on orbit solvers from James Graham and Rob De Rosa. Adapted by Jason Wang.

    Args:
        epochs (np.array): MJD times for which we want the positions of the planet
        sma (np.array): semi-major axis of orbit [au]
        ecc (np.array): eccentricity of the orbit [0,1]
        tau (np.array): epoch of periastron passage in fraction of orbital period past MJD=0 [0,1]
        argp (np.array): argument of periastron [radians]
        lan (np.array): longitude of the ascending node [radians]
        inc (np.array): inclination [radians]
        plx (np.array): parallax [mas]
        mtot (np.array): total mass [Solar masses]
        mass (np.array): mass of the body [Solar masses]. For planets mass ~ 0 (default)

    Return:
        raoff (np.array): 2-D array (n_orbs x n_dates) of RA offsets between the bodies (origin is at the other body)
        deoff (np.array): 2-D array (n_orbs x n_dates) of Dec offsets between the bodies
        vz (np.array): 2-D array (n_orbs x n_dates) of radial velocity offset between the bodies

    Written: Jason Wang, Henry Ngo, 2018
    """

    n_orbs  = np.size(sma)  # num sets of input orbital parameters
    n_dates = np.size(epochs) # number of dates to compute offsets and vz

    # Necessary for _calc_ecc_anom, for now
    if np.isscalar(epochs): # just in case epochs is given as a scalar
        epochs = np.array([epochs])
    ecc_arr = np.tile(ecc, (n_dates, 1))

    # If mass not given, assume test particle case
    if mass is None:
        mass = np.zeros(n_orbs)

    # Compute period (from Kepler's third law) and mean motion
    period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
    period = period.to(u.day).value
    mean_motion = 2*np.pi/(period) # in rad/day

    # # compute mean anomaly (size: n_orbs x n_dates)
    manom = (mean_motion*epochs[:, None] - 2*np.pi*tau) % (2.0*np.pi)

    # compute eccentric anomalies (size: n_orbs x n_dates)
    eanom = _calc_ecc_anom(manom, ecc_arr)

    # compute the true anomalies (size: n_orbs x n_dates)
    # Note: matrix multiplication makes the shapes work out here and below
    tanom = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eanom) )
    # compute 3-D orbital radius of second body (size: n_orbs x n_dates)
    radius = sma * (1.0 - ecc * np.cos(eanom))

    # compute ra/dec offsets (size: n_orbs x n_dates)
    # math from James Graham. Lots of trig
    c2i2 = np.cos(0.5*inc)**2
    s2i2 = np.sin(0.5*inc)**2
    arg1 = tanom + argp + lan
    arg2 = tanom + argp - lan
    c1 = np.cos(arg1)
    c2 = np.cos(arg2)
    s1 = np.sin(arg1)
    s2 = np.sin(arg2)

    # updated sign convention for Green Eq. 19.4-19.7
    # return values in arcsecons
    plx_as = plx * 1e-3

    raoff = radius * (c2i2*s1 - s2i2*s2) * plx_as
    deoff = radius * (c2i2*c1 + s2i2*c2) * plx_as

    # compute the radial velocity (vz) of the body (size: n_orbs x n_dates)
    # first comptue the RV semi-amplitude (size: n_orbs x n_dates)
    m1 = mtot - mass # mass of the primary star
    Kv = np.sqrt(consts.G / (1.0 - ecc**2)) * (m1 * u.Msun * np.sin(inc)) / np.sqrt(mtot * u.Msun) / np.sqrt(sma * u.au)
    # Convert to km/s
    Kv = Kv.to(u.km/u.s)
    # compute the vz
    vz =  Kv.value * ( ecc*np.cos(argp) + np.cos(argp + tanom) )

    # Squeeze out extra dimension (useful if n_orbs = 1, does nothing if n_orbs > 1)
    # [()] used to convert 1-element arrays into scalars, has no effect for larger arrays
    raoff = np.transpose(np.squeeze(raoff)[()])
    deoff = np.transpose(np.squeeze(deoff)[()])
    vz = np.transpose(np.squeeze(vz)[()])

    return raoff, deoff, vz

def _calc_ecc_anom(manom, ecc, tolerance=1e-9, max_iter=100):
    """
    Computes the eccentric anomaly from the mean anomlay.
    Code from Rob De Rosa's orbit solver (e < 0.95 use Newton, e >= 0.95 use Mikkola)

    Args:
        manom (float/np.array): mean anomaly, either a scalar or np.array of any shape
        ecc (float/np.array): eccentricity, either a scalar or np.array of the same shape as manom
        tolerance (float, optional): absolute tolerance of iterative computation. Defaults to 1e-9.
        max_iter (int, optional): maximum number of iterations before switching. Defaults to 100.
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
    if len(ind_zero[0]) > 0: eanom[ind_zero] = manom[ind_zero]

    # Now low eccentricities
    ind_low = np.where(~ecc_zero & ecc_low)
    if len(ind_low[0]) > 0: eanom[ind_low] = _newton_solver(manom[ind_low], ecc[ind_low], tolerance=tolerance, max_iter=max_iter)

    # Now high eccentricities
    ind_high = np.where(~ecc_zero & ~ecc_low)
    if len(ind_high[0]) > 0: eanom[ind_high] = _mikkola_solver_wrapper(manom[ind_high], ecc[ind_high])

    return np.squeeze(eanom)[()]

def _newton_solver(manom, ecc, tolerance=1e-9, max_iter=100, eanom0=None):
    """
    Newton-Raphson solver for eccentric anomaly.
    Args:
        manom (np.array): array of mean anomalies
        ecc (np.array): array of eccentricities
        eanom0 (np.array): array of first guess for eccentric anomaly, same shape as manom (optional)
    Return:
        eanom (np.array): array of eccentric anomalies

    Written: Rob De Rosa, 2018

    """

    # Initialize at E=0, E=pi is better at very high eccentricities
    if eanom0 is None:
        eanom = np.full(np.shape(manom), 0.0)
    else:
        eanom = np.copy(eanom0)

    # Let's do one iteration to start with
    eanom -= (eanom - (ecc * np.sin(eanom)) - manom) / (1.0 - (ecc * np.cos(eanom)))

    diff = (eanom - (ecc * np.sin(eanom)) - manom) / (1.0 - (ecc * np.cos(eanom)))
    abs_diff = np.abs(diff)
    ind = np.where(abs_diff > tolerance)
    niter = 0
    while ((ind[0].size > 0) and (niter <= max_iter)):
        eanom[ind] -= diff[ind]
        # If it hasn't converged after half the iterations are done, try starting from pi
        if niter == (max_iter//2):
            eanom[ind] = np.pi
        diff[ind] = (eanom[ind] - (ecc[ind] * np.sin(eanom[ind])) - manom[ind]) / (1.0 - (ecc[ind] * np.cos(eanom[ind])))
        abs_diff[ind] = np.abs(diff[ind])
        ind = np.where(abs_diff > tolerance)
        niter += 1

    if niter >= max_iter:
        print(manom[ind], eanom[ind], diff[ind], ecc[ind], '> {} iter.'.format(max_iter))
        eanom[ind] = _mikkola_solver_wrapper(manom[ind], ecc[ind]) # Send remaining orbits to the analytical version, this has not happened yet...

    return eanom

def _mikkola_solver_wrapper(manom, ecc):
    """
    Analtyical Mikkola solver (S. Mikkola. 1987. Celestial Mechanics, 40 , 329-334.) for the eccentric anomaly.
    Wrapper for the python implemenation of the IDL version. From Rob De Rosa.

    Args:
        manom (np.array): array of mean anomalies between 0 and 2pi
        ecc (np.array): eccentricity
    Return:
        eanom (np.array): array of eccentric anomalies

    Written: Jason Wang, 2018
    """

    ind_change = np.where(manom > np.pi)
    manom[ind_change] = (2.0 * np.pi) - manom[ind_change]
    eanom = _mikkola_solver(manom, ecc)
    eanom[ind_change] = (2.0 * np.pi) - eanom[ind_change]

    return eanom

def _mikkola_solver(manom, ecc):
    """
    Analtyical Mikkola solver for the eccentric anomaly.
    Adapted from IDL routine keplereq.pro by Rob De Rosa http://www.lpl.arizona.edu/~bjackson/idl_code/keplereq.pro

    Args:
        manom (float or np.array): mean anomaly, must be between 0 and pi.
        ecc (float or np.array): eccentricity
    Return:
        eanom (np.array): array of eccentric anomalies

    Written: Jason Wang, 2018
    """

    alpha = (1.0 - ecc) / ((4.0 * ecc) + 0.5)
    beta = (0.5 * manom) / ((4.0 * ecc) + 0.5)

    aux = np.sqrt(beta**2.0 + alpha**3.0)
    z = np.abs(beta + aux)**(1.0/3.0)

    s0 = z - (alpha/z)
    s1 = s0 - (0.078*(s0**5.0)) / (1.0 + ecc)
    e0 = manom + (ecc * (3.0*s1 - 4.0*(s1**3.0)))

    se0=np.sin(e0)
    ce0=np.cos(e0)

    f  = e0-ecc*se0-manom
    f1 = 1.0-ecc*ce0
    f2 = ecc*se0
    f3 = ecc*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+(1.0/6.0)*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+(1.0/6.0)*f3*u3*u3+(1.0/24.0)*f4*(u3**3.0))

    return (e0 + u4)
