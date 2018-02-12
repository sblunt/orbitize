"""
This module solves for the orbit of the planet given Keplerian parameters
"""
import numpy as np
import astropy.units as u
import astropy.constants as consts


def calc_orbit(epochs, sma, ecc, tau, argp, lan, inc, plx, mtot, mass=0):
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
        mtot (np.array): total mass [Solar masses]. Note that this is
        mass (float): mass of this body [Solar masses]. For planets mass ~ 0

    Return:
        raoff (np.array): 2-D array (n_orbs x n_dates) of RA offsets between the bodies (origin is at the other body)
        deoff (np.array): 2-D array (n_orbs x n_dates) of Dec offsets between the bodies
        vz (np.array): 2-D array (n_orbs x n_dates) of radial velocity offset between the bodies

    Written: Jason Wang, Henry Ngo, 2018
    """

    n_orbs  = np.size(sma)  # num sets of input orbital parameters
    n_dates = np.size(epochs) # number of dates to compute offsets and vz

    # Compute period (from Kepler's third law) and mean motion
    period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
    period = period.to(u.day).value
    mean_motion = 2*np.pi/(period) # in rad/day

    # compute mean anomaly (size: n_orbs x n_dates)
    manom = np.zeros((n_orbs,n_dates))
    for jj in range(0,n_dates):
        manom[:,jj] = mean_motion*epochs[jj] - 2*np.pi*tau

    # compute eccentric anomalies (size: n_orbs x n_dates)
    eanom = np.zeros((n_orbs,n_dates))
    if n_orbs > 1:
        for ii in range(0,n_orbs):
            eanom[ii,:] = _calc_ecc_anom(manom[ii,:], ecc[ii])
    else:
        eanom[0,:] = _calc_ecc_anom(manom[0,:], ecc)

    # compute the true anomalies (size: n_orbs x n_dates)
    tanom = np.zeros((n_orbs,n_dates))
    for jj in range(0,n_dates):
        tanom[:,jj] = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eanom[:,jj]) )
    # compute 3-D orbital radius of second body (size: n_orbs x n_dates)
    radius = np.zeros((n_orbs,n_dates))
    for jj in range(0,n_dates):
        radius[:,jj] = sma * (1.0 - ecc * np.cos(eanom[:,jj]))

    # compute ra/dec offsets (size: n_orbs x n_dates)
    raoff = np.zeros((n_orbs,n_dates))
    deoff = np.zeros((n_orbs,n_dates))
    # math from James Graham. Lots of trig
    c2i2 = np.cos(0.5*inc)**2
    s2i2 = np.sin(0.5*inc)**2
    for jj in range(0,n_dates):
        arg0 = tanom[:,jj] + lan
        arg1 = tanom[:,jj] + argp + lan
        arg2 = tanom[:,jj] + argp - lan
        arg3 = tanom[:,jj] - lan

        c1 = np.cos(arg1)
        c2 = np.cos(arg2)
        s1 = np.sin(arg1)
        s2 = np.sin(arg2)
        sa0 = np.sin(arg0)
        sa3 = np.sin(arg3)

        # updated sign convention for Green Eq. 19.4-19.7
        # return values in arcsecons
        plx_as = plx * 1e-3

        raoff[:,jj] = radius[:,jj] * (c2i2*s1 - s2i2*s2) * plx_as
        deoff[:,jj] = radius[:,jj] * (c2i2*c1 + s2i2*c2) * plx_as

    # compute the radial velocity (vz) of the body (size: n_orbs x n_dates)
    vz = np.zeros((n_orbs,n_dates))
    # first comptue the RV semi-amplitude (size: n_orbs)
    # Treat entries where mass = 0 (test particle) and massive bodies separately
    if mass == 0:
        # basically treating this body as a test particle. we can calcualte a radial velocity for a test particle
        Kv =  mean_motion * (sma * np.sin(inc)) / np.sqrt(1 - ecc**2) * (u.au/u.day)
        Kv = Kv.to(u.km/u.s) # converted to km/s
    else:
        # we want to measure the mass of the influencing body on the system
        # we need units now
        m2 = mtot - mass
        Kv = np.sqrt(consts.G / (1.0 - ecc**2)) * (m2 * u.Msun * np.sin(inc)) / np.sqrt(mtot * u.Msun) / np.sqrt(sma * u.au)
        Kv = Kv.to(u.km/u.s)
    # compute the vz
    for jj in range(0,n_dates):
        vz[:,jj] =  Kv.value * ( ecc*np.cos(argp) + np.cos(argp + tanom[:,jj]) )

    # Squeeze out extra dimension (useful if n_orbs = 1, does nothing if n_orbs > 1)
    raoff = np.squeeze(raoff)
    deoff = np.squeeze(deoff)
    vz = np.squeeze(vz)

    return raoff, deoff, vz


def _calc_ecc_anom(manom, ecc, tolerance=1e-9, max_iter=100):
    """
    Computes the eccentric anomaly from the mean anomlay.
    Code from Rob De Rosa's orbit solver (e < 0.95 use Newton, e >= 0.95 use Mikkola)

    Args:
        manom (np.array): array of mean anomalies
        ecc (float): eccentricity
        tolerance (float, optional): absolute tolerance of iterative computation. Defaults to 1e-9.
        max_iter (int, optional): maximum number of iterations before switching. Defaults to 100.
    Return:
        eanom (np.array): array of eccentric anomalies

    Written: Jason Wang, 2018
    """

    if ecc == 0.0:
        eanom = np.copy(manom)
    else:
        if ecc < 0.95:
            eanom = np.copy(manom)

            #Let's do two iterations to start with
            eanom -= (eanom - (ecc * np.sin(eanom)) - manom) / (1.0 - (ecc * np.cos(eanom)))
            eanom -= (eanom - (ecc * np.sin(eanom)) - manom) / (1.0 - (ecc * np.cos(eanom)))

            diff = (eanom - (ecc * np.sin(eanom)) - manom) / (1.0 - (ecc * np.cos(eanom)))
            abs_diff = np.abs(diff)
            ind = np.where(abs_diff > tolerance)
            niter = 0
            while ((ind[0].size > 0) and (niter <= max_iter)):
                eanom[ind] -= diff[ind]
                diff[ind] = (eanom[ind] - (ecc * np.sin(eanom[ind])) - manom[ind]) / (1.0 - (ecc * np.cos(eanom[ind])))
                abs_diff[ind] = np.abs(diff[ind])
                ind = np.where(abs_diff > tolerance)
                niter += 1
            if niter >= max_iter:
                print(manom[ind], eanom[ind], ecc, '> {} iter.'.format(max_iter))
                eanom = _mikkola_solver_wrapper(manom, ecc)
        else:
            eanom = _mikkola_solver_wrapper(manom, ecc) # Send it to the analytical version, this has not happened yet...

    return eanom

def _mikkola_solver_wrapper(manom, e):
    """
    Analtyical Mikkola solver (S. Mikkola. 1987. Celestial Mechanics, 40 , 329-334.) for the eccentric anomaly.
    Wrapper for the python implemenation of the IDL version. From Rob De Rosa.

    Args:
        manom (np.array): array of mean anomalies
        ecc (float): eccentricity
    Return:
        eccanom (np.array): array of eccentric anomalies

    Written: Jason Wang, 2018
    """
    ind_change = np.where(manom > np.pi)
    manom[ind_change] = (2.0 * np.pi) - manom[ind_change]
    Eanom = _mikkola_solver(manom, e)
    Eanom[ind_change] = (2.0 * np.pi) - Eanom[ind_change]

    return Eanom

def _mikkola_solver(manom, e):
    """
    Analtyical Mikkola solver for the eccentric anomaly.
    Adapted from IDL routine keplereq.pro by Rob De Rosa http://www.lpl.arizona.edu/~bjackson/idl_code/keplereq.pro

    Args:
        manom (np.array): array of mean anomalies
        ecc (float): eccentricity
    Return:
        eccanom (np.array): array of eccentric anomalies

    Written: Jason Wang, 2018
    """

    alpha = (1.0 - e) / ((4.0 * e) + 0.5)
    beta = (0.5 * manom) / ((4.0 * e) + 0.5)

    aux = np.sqrt(beta**2.0 + alpha**3.0)
    z = beta + aux
    z = z**(1.0/3.0)

    s0 = z - (alpha/z)
    s1 = s0 - (0.078*(s0**5.0)) / (1.0 + e)
    e0 = manom + (e * (3.0*s1 - 4.0*(s1**3.0)))

    se0=np.sin(e0)
    ce0=np.cos(e0)

    f  = e0-e*se0-manom
    f1 = 1.0-e*ce0
    f2 = e*se0
    f3 = e*ce0
    f4 = -f2
    u1 = -f/f1
    u2 = -f/(f1+0.5*f2*u1)
    u3 = -f/(f1+0.5*f2*u2+0.16666666666667*f3*u2*u2)
    u4 = -f/(f1+0.5*f2*u3+0.16666666666667*f3*u3*u3+0.041666666666667*f4*(u3**3.0))

    return (e0 + u4)
