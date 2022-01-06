import numpy as np
import orbitize.basis as basis
import rebound

def calc_orbit(
    epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, tau_ref_epoch=58849, 
    m_pl=None, output_star=False
):
    """
    Solves for position for a set of input orbital elements using rebound.

    Args:
        epochs (np.array): MJD times for which we want the positions of the planet
        sma (np.array): semi-major axis of orbit [au]
        ecc (np.array): eccentricity of the orbit [0,1]
        inc (np.array): inclination [radians]
        aop (np.array): argument of periastron [radians]
        pan (np.array): longitude of the ascending node [radians]
        tau (np.array): epoch of periastron passage in fraction of orbital period 
            past MJD=0 [0,1]
        plx (np.array): parallax [mas]
        mtot (np.array): total mass of the two-body orbit (M_* + M_planet) 
            [Solar masses]
        tau_ref_epoch (float, optional): reference date that tau is defined with
            respect to
        m_pl (np.array, optional): mass of the planets[units]
        output_star (bool): if True, also return the position of the star
            relative to the barycenter.

    Returns:
        3-tuple:

            raoff (np.array): array-like (n_dates x n_orbs) of RA offsets between 
                the bodies (origin is at the other body) [mas]

            deoff (np.array): array-like (n_dates x n_orbs) of Dec offsets between 
                the bodies [mas]
                
            vz (np.array): array-like (n_dates x n_orbs) of radial velocity of 
                one of the bodies (see `mass_for_Kamp` description)  [km/s]

    .. Note:: 
    
        return is in format [raoff[planet1, planet2,...,planetn], 
        deoff[planet1, planet2,...,planetn], vz[planet1, planet2,...,planetn]
    """
    
    sim = rebound.Simulation()      #creating the simulation in Rebound
    sim.units = ('AU','yr','Msun')  #converting units for uniformity
    sim.G = 39.476926408897626      #Using a more accurate value in order to minimize differences from prev. Kepler solver
    ps = sim.particles              #for easier calls

    tx = len(epochs)                #keeping track of how many time steps
    te = epochs-epochs[0]           #days

    indv = len(sma)                 #number of planets orbiting the star
    num_planets = np.arange(0,indv) #creates an array of indeces for each planet that exists

    if m_pl is None:                #if no planet masses are input, planet masses set ot zero and mass of star is equal to mtot
        sim.add(m = mtot)
        m_pl = np.zeros(len(sma))
        m_star = mtot
    else:                           #mass of star is always (mass of system)-(sum of planet masses) 
        m_star = mtot - sum(m_pl)
        sim.add(m = m_star)

    #for each planet, create a body in the Rebound sim
    for i in num_planets:
        #calculating mean anomaly
        m_interior = m_star + sum(m_pl[0:i+1])
        mnm = basis.tau_to_manom(epochs[0], sma[i], m_interior, tau[i], tau_ref_epoch) 
        #adding each planet
        sim.add(m = m_pl[i], a = sma[i], e = ecc[i], inc = inc[i], Omega = pan[i] + np.pi/2, omega =aop[i], M =mnm)

    sim.move_to_com()
    sim.integrator = "ias15" 
    sim.dt = ps[1].P/100.       #good rule of thumb: timestep should be at most 10% of the shortest orbital period

    if output_star:
        ra_reb = np.zeros((tx, indv+1))   #numpy.zeros(number of [arrays], size of each array)
        dec_reb = np.zeros((tx, indv+1))
        vz = np.zeros((tx, indv+1))
        for j,t in enumerate(te):
            sim.integrate(t/365.25)
            #for the star and each planet in each epoch denoted by j,t find the RA, Dec, and RV
            com = sim.calculate_com()
            ra_reb[j,0] = -(ps[0].x-com.x)# ra is negative x
            dec_reb[j,0] = ps[0].y-com.y
            vz[j,0] = ps[0].vz
            for i in num_planets:
                ra_reb[j,i+1] = -(ps[int(i+1)].x - ps[0].x) # ra is negative x
                dec_reb[j,i+1] = ps[int(i+1)].y - ps[0].y
                vz[j,i+1] = ps[int(i+1)].vz
    else:
        ra_reb = np.zeros((tx, indv))   #numpy.zeros(number of [arrays], size of each array)
        dec_reb = np.zeros((tx, indv))
        vz = np.zeros((tx, indv))
        #integrate at each epoch
        for j,t in enumerate(te):
            sim.integrate(t/365.25)
            #for each planet in each epoch denoted by j,t find the RA, Dec, and RV
            for i in num_planets:
                ra_reb[j,i] = -(ps[int(i+1)].x - ps[0].x) # ra is negative x
                dec_reb[j,i] = ps[int(i+1)].y - ps[0].y
                vz[j,i] = ps[int(i+1)].vz
    

    #adjusting for parallax
    raoff = plx*ra_reb
    deoff = plx*dec_reb

    #for formatting purposes
    if len(sma)==1:
        raoff = raoff.reshape(tx,)
        deoff = deoff.reshape(tx,)
        vz = vz.reshape(tx,)
        return raoff, deoff, vz
    else: 
        return raoff, deoff, vz