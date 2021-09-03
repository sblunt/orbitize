import numpy as np
from astropy import units as u
from astropy import constants as consts
from orbitize import kepler

"""
This module converts orbital elements between different coordinate bases.
"""

def ecosomega_to_standard(elems):
    """
    Converts array of orbital elements with ecc*cos aop and ecc*sin aop to the usual one with ecc and aop.

    Args:
        elems (np.array of floats): Orbital elements with e cosine omega and e sin omega (semi-major axis [au], ecc*cos aop,
            inclination [radians], ecc*sin aop, longitude of the ascending node [radians], epoch of 
            periastron passage in fraction of orbital period past MJD=0 [0,1], parallax [mas], total mass of the two-body orbit
            (M_* + M_planet) [Solar masses]). If more than 1 set of parameters is passed, the first dimension must be
            the number of orbital parameter sets, and the second the orbital elements.

    Return:
        np.array: Orbital elements with ecc and aop
    """
    if elems.ndim == 1:
        elems = elems[np.newaxis, :]
    ecos, esin = elems[:,1], elems[:,3]
    # Solve for aop, in the interval [-pi, pi]
    aop = np.arctan2(esin, ecos)
    # Fix interval to [0, 2 pi]
    aop[aop < 0.0] = 2*np.pi + aop[aop < 0.0]
    #Solve for ecc, taking care of possible negative cosines
    ecc = np.sqrt(ecos**2 / (np.cos(aop)**2))

    elems[:,1], elems[:,3] = ecc, aop

    return np.squeeze(elems)

def standard_to_ecosomega(elems):
    """
    Converts array of orbital elements from the usual basis to ecc*cos aop and ecc*sin aop.

    Args:
        elems (np.array of floats): Orbital elements with e cosine omega and e sin omega (semi-major axis [au], ecc,
            inclination [radians], aop, longitude of the ascending node [radians], epoch of 
            periastron passage in fraction of orbital period past MJD=0 [0,1], parallax [mas], total mass of the two-body orbit
            (M_* + M_planet) [Solar masses]). If more than 1 set of parameters is passed, the first dimension must be
            the number of orbital parameter sets, and the second the orbital elements.

    Return:
        np.array: Orbital elements with ecc*cos aop and ecc*sin aop at indices 1 and 3.
    """
    if elems.ndim == 1:
        elems = elems[np.newaxis, :]
    ecc, aop = elems[:,1], elems[:,3]
    e_sin, e_cos = ecc*np.sin(aop), ecc*np.cos(aop)

    elems[:,1], elems[:,3] = e_cos, e_sin

    return np.squeeze(elems)

def xyz_to_standard(epoch, elems, tau_ref_epoch=58849):
    """
    Converts array of orbital elements in terms of position and velocity in xyz to the regular one

    Args:
        epoch (float): Date in MJD of observation to calculate time of periastron passage (tau).
        elems (np.array of floats): Orbital elements in xyz basis (x-coordinate [au], y-coordinate [au],
            z-coordinate [au], velocity in x [km/s], velocity in y [km/s], velocity in z [km/s], parallax [mas], total mass of the two-body orbit
            (M_* + M_planet) [Solar masses]). If more than 1 set of parameters is passed, the first dimension must be
            the number of orbital parameter sets, and the second the orbital elements.

    Return:
        np.array: Orbital elements in the usual basis (sma, ecc, inc, aop, pan, tau, plx, mtot)
    """

    if elems.ndim == 1:
        elems = elems[np.newaxis, :]
    # Velocities and positions, with units
    vel = elems[:,3:6] * u.km / u.s # velocities in km / s ?
    pos = elems[:,0:3] * u.AU # positions in AU ?
    vel_magnitude = np.linalg.norm(vel, axis=1)
    pos_magnitude = np.linalg.norm(pos, axis=1)

    # Mass
    mtot = elems[:,7]*u.Msun
    mu = consts.G * mtot # G in m3 kg-1 s-2, mtot in msun

    # Angular momentum, making sure nodal vector is not exactly zero
    h = (np.cross(pos, vel, axis=1)).si
    # if h[0].value == 0.0 and h[1].value == 0.0:
    #     pos[2] = 1e-8*u.AU
    #     h = (np.cross(pos, vel)).si
    h_magnitude = np.linalg.norm(h, axis=1)

    sma = 1 / (2.0 / pos_magnitude - (vel_magnitude**2)/mu)
    sma = sma.to(u.AU)

    ecc = (np.sqrt(1 - h_magnitude**2 / (sma * mu))).value
    e_vector = (np.cross(vel, h, axis=1) / mu[:, None] - pos / pos_magnitude[:, None]).si
    e_vec_magnitude = np.linalg.norm(e_vector, axis=1)

    unit_k = np.array((0,0,1))
    cos_inc = (np.dot(h, unit_k) / h_magnitude).value
    inc = np.arccos(-cos_inc)

    #Nodal vector
    n = np.cross(unit_k, h)
    n_magnitude = np.linalg.norm(n, axis=1)

    # Position angle of the nodes, checking for the right quadrant
    # np.arccos yields angles in [0, pi]
    unit_i = np.array((1,0,0))
    unit_j = np.array((0,1,0))
    cos_pan = (np.dot(n, unit_j) / n_magnitude).value
    pan = np.arccos(cos_pan)
    n_x = np.dot(n, unit_i)
    pan[n_x < 0.0] = 2*np.pi - pan[n_x < 0.0]

    # Argument of periastron, checking for the right quadrant
    cos_aop = (np.sum(n*e_vector, axis=1) / (n_magnitude * e_vec_magnitude)).value
    aop = np.arccos(cos_aop)
    e_vector_z = np.dot(e_vector, unit_k)
    aop[e_vector_z < 0.0] = 2.0*np.pi - aop[e_vector_z < 0.0]

    # True anomaly, checking for the right quadrant
    cos_tanom = (np.sum(pos*e_vector, axis=1) / (pos_magnitude*e_vec_magnitude)).value
    tanom = np.arccos(cos_tanom)
    # Check for places where tanom is nan, due to cos_tanom=1. (for some reason that was a problem)
    tanom = np.where((0.9999<cos_tanom ) & (cos_tanom<1.001), 0.0, tanom)
    rdotv = np.sum(pos*vel, axis=1)
    tanom[rdotv < 0.0] = 2*np.pi - tanom[rdotv < 0.0]

    # Eccentric anomaly to get tau, checking for the right quadrant
    cos_eanom = ((1 - pos_magnitude / sma) / ecc).value
    eanom = np.arccos(cos_eanom)
    # Check for places where eanom is nan, due to cos_eanom = 1.(same problem as above)
    eanom = np.where((0.9999<cos_eanom ) & (cos_eanom<1.001), 0.0, eanom)
    eanom[tanom > np.pi] =  2*np.pi - eanom[tanom > np.pi]

    # Time of periastron passage, using Kepler's equation, in MJD:
    time_tau = epoch - ((np.sqrt(sma**3 / mu)).to(u.day)).value * (eanom - ecc*np.sin(eanom))

    # Calculate period from Kepler's third law, in days:
    period = np.sqrt(4*np.pi**2.0*(sma)**3/mu)
    period = period.to(u.day).value

    # Finally, tau
    tau = (time_tau - tau_ref_epoch) / period
    tau = tau%1.0

    mtot = mtot.value
    sma = sma.value

    results = np.array((sma, ecc, inc, aop, pan, tau, elems[:,6], mtot)).T

    return np.squeeze(results)

def standard_to_xyz(epoch, elems, tau_ref_epoch=58849, tolerance=1e-9, max_iter=100):
    """
    Converts array of orbital elements from the regular base of Keplerian orbits to positions and velocities in xyz
    Uses code from orbitize.kepler

    Args:
        epoch (float): Date in MJD of observation to calculate time of periastron passage (tau).
        elems (np.array of floats): Orbital elements (sma, ecc, inc, aop, pan, tau, plx, mtot).
                If more than 1 set of parameters is passed, the first dimension must be
                the number of orbital parameter sets, and the second the orbital elements.

    Return:
        np.array: Orbital elements in xyz (x-coordinate [au], y-coordinate [au], z-coordinate [au], 
        velocity in x [km/s], velocity in y [km/s], velocity in z [km/s], parallax [mas], total mass of the two-body orbit
            (M_* + M_planet) [Solar masses])
    """

    # Use classical elements to obtain position and velocity in the perifocal coordinate system
    # Then transform coordinates using matrix multiplication

    if elems.ndim == 1:
        elems = elems[np.newaxis, :]
    # Define variables
    sma = elems[:,0] # AU
    ecc = elems[:,1] # [0.0, 1.0]
    inc = elems[:,2] # rad [0, pi]
    aop = elems[:,3] # rad [0, 2 pi]
    pan = elems[:,4] # rad [0, 2 pi]
    tau = elems[:,5] # [0.0, 1.0]
    mtot = elems[:,7] # Msun

    # Just in case so nothing breaks
    ecc = np.where(ecc == 0.0, 1e-8, ecc)
    inc = np.where(inc == 0.0, 1e-8, inc)

    # Begin by calculating the eccentric anomaly
    period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
    period = period.to(u.day).value # Period in days
    mean_motion = 2*np.pi/(period)

    # Mean anomaly
    manom = (mean_motion*(epoch - tau_ref_epoch) - 2*np.pi*tau) % (2.0*np.pi)
    # Eccentric anomaly
    eanom = kepler._calc_ecc_anom(manom, ecc, tolerance=tolerance, max_iter=max_iter)
    # if eanom.ndim == 1:
    #     eanom = eanom[np.newaxis, :]
    # Magnitude of angular momentum:
    h = np.sqrt(consts.G*(mtot*u.Msun)*(sma*u.AU)*(1 - ecc**2))

    # Position vector in the perifocal system in AU
    pos_peri_x = (sma*(np.cos(eanom) - ecc))
    pos_peri_y = (sma*np.sqrt(1 - ecc**2)*np.sin(eanom))
    pos_peri_z = np.zeros(len(elems))

    pos = np.stack((pos_peri_x, pos_peri_y, pos_peri_z)).T
    pos_magnitude = np.linalg.norm(pos, axis=1)

    # Velocity vector in the perifocal system in km/s
    vel_peri_x = - ((np.sqrt(consts.G*(mtot*u.Msun)*(sma*u.AU))*np.sin(eanom) / (pos_magnitude*u.AU)).to(u.km / u.s)).value 
    vel_peri_y = ((h* np.cos(eanom) / (pos_magnitude*u.AU)).to(u.km / u.s)).value
    vel_peri_z = np.zeros(len(elems))

    vel = np.stack((vel_peri_x, vel_peri_y, vel_peri_z)).T

    # Transformation matrix to inertial xyz system, component by component
    pan = pan +np.pi / 2.0
    T_11 = np.cos(pan)*np.cos(aop) - np.sin(pan)*np.sin(aop)*np.cos(inc)
    T_12 = - np.cos(pan)*np.sin(aop) - np.sin(pan)*np.cos(aop)*np.cos(inc)
    T_13 = np.sin(pan)*np.sin(inc)

    T_21 = np.sin(pan)*np.cos(aop) + np.cos(pan)*np.sin(aop)*np.cos(inc)
    T_22 = - np.sin(pan)*np.sin(aop) + np.cos(pan)*np.cos(aop)*np.cos(inc)
    T_23 = - np.cos(pan)*np.sin(inc)

    T_31 = np.sin(aop)*np.sin(inc)
    T_32 = np.cos(aop)*np.sin(inc)
    T_33 = np.cos(inc)

    T = np.array([[T_11, T_12, T_13],
                  [T_21, T_22, T_23],
                  [T_31, T_32, T_33]])

    pos_xyz = np.zeros((len(elems), 3))
    vel_xyz = np.zeros((len(elems), 3))
    for k in range(len(elems)):
        pos_xyz[k,:] =  np.matmul(T[:,:,k], pos[k])
        vel_xyz[k,:] =  np.matmul(T[:,:,k], vel[k])

    # Flipping x-axis sign to increase X as RA increases
    result = np.stack([-pos_xyz[:,0], pos_xyz[:,1], pos_xyz[:,2], -vel_xyz[:,0], vel_xyz[:,1], vel_xyz[:,2], elems[:,6], mtot]).T

    return np.squeeze(result)
