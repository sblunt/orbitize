import os
import numpy as np
import astropy
import astropy.units as u
import orbitize.basis as basis
import orbitize.kepler
import rebound
import matplotlib.pyplot as plt


def calc_orbit(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, tau_ref_epoch, m_pl=None):
    """
    Solves for position for a set of input orbital elements using rebound.
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
        tau_ref_epoch (float, optional): reference date that tau is defined with respect to (i.e., tau=0)
        m_pl (np.array, optional): mass of the planets[units]
    Returns:
        3-tuple:
            raoff (np.array): array-like (n_dates x n_orbs) of RA offsets between the bodies
            (origin is at the other body) [mas]
            deoff (np.array): array-like (n_dates x n_orbs) of Dec offsets between the bodies [mas]
            vz (np.array): array-like (n_dates x n_orbs) of radial velocity of one of the bodies
                (see `mass_for_Kamp` description)  [km/s]
        return is in format [raoff[planet1, planet2,...,planetn], deoff[planet1, planet2,...,planetn], vz[planet1, planet2,...,planetn]
    """
    
    sim = rebound.Simulation()      #creating the simulation in Rebound
    sim.units = ('AU','yr','Msun')  #converting units for uniformity
    sim.G = 39.476926408897626      #Using a more accurate value in order to minimize differences from prev. Kepler solver
    ps = sim.particles              #for easier calls

    tx = len(epochs)                #keeping track of how many time steps
    te = epochs-epochs[0]           #days

    indv = len(sma)                 #number of planets orbiting the star
    num_planets = np.arange(0,indv) #creates an array of indeces for each planet that exists
    ra_reb = np.zeros((tx, indv))   #numpy.zeros(number of [arrays], size of each array)
    dec_reb = np.zeros((tx, indv))
    vz = np.zeros((tx, indv))

    if m_pl is None:                #if no planet masses are input, planet masses set ot zero and mass of star is equal to mtot
        sim.add(m = mtot)
        m_pl = np.zeros(len(sma))

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

    

#Test Data
import astropy.units as u
mass_in_mjup = 10
mB_Jup = 7
mass_in_msun = mass_in_mjup * u.Mjup.to(u.Msun)
massB = mB_Jup * u.Mjup.to(u.Msun)
m_pl = np.array([mass_in_msun, mass_in_msun, mass_in_msun, massB])

#From System HR-8799
#NOTE planets get closer to the star in alphabetical order, i.e. B is farthest, E is closest
sma = np.array([16, 26,43, 71]) 
ecc = np.array([.12, .13, .02, .02])
inc = np.array([0.47123, 0.47123, 0.47123, 0.47123])
aop = np.array([1.91986, 0.29670, 1.16937, 1.5184])
pan = np.array([1.18682, 1.18682, 1.18682, 1.18682])
tau = np.array([0.71, 0.79, 0.50, 0.54])
plx = np.array([7])
mtot = np.array([1.49])
tau_ref_epoch = 0
years = 365.25*5
#epoch = np.linspace(0,years,1000)

import orbitize.kepler
from astropy.time import Time
from orbitize import DATADIR
from orbitize.system import System
from orbitize.read_input import read_file

# need a properly formatted data table for the code to run but it doesn't 
# matter what's in it
input_file = '{}/GJ504.csv'.format(DATADIR)
data_table = read_file(input_file)
num_secondary_bodies = 4

epochs = Time(np.linspace(2020, 2025, num=int(1000)), format='decimalyear').mjd

sma1 = sma[0]
ecc1 = ecc[0]
inc1 = inc[0]
aop1 = aop[0]
pan1 = pan[0]
tau1 = tau[0]
sma2 = sma[1]
ecc2 = ecc[1]
inc2 = inc[1]
aop2 = aop[1]
pan2 = pan[1]
tau2 = tau[1]
sma3 = sma[2]
ecc3 = ecc[2]
inc3 = inc[2]
aop3 = aop[2]
pan3 = pan[2]
tau3 = tau[2]
sma4 = sma[3]
ecc4 = ecc[3]
inc4 = inc[3]
aop4 = aop[3]
pan4 = pan[3]
tau4 = tau[3]
m1 = mass_in_msun
m2 = mass_in_msun
m3 = mass_in_msun
m4 = massB
m_st = mtot-sum(m_pl)

hr8799_sys = System(
    num_secondary_bodies, data_table, m_st,
    plx, fit_secondary_mass=True, tau_ref_epoch=tau_ref_epoch
)

params_arr = np.array([
    sma1, ecc1, inc1, aop1, pan1, tau1,
    sma2, ecc2, inc2, aop2, pan2, tau2,
    sma3, ecc3, inc3, aop3, pan3, tau3,
    sma4, ecc4, inc4, aop4, pan4, tau4,
    plx,
    m1, m2, m3, m4, m_st
])

# these arrays have shape (n_epochs x n_bodies x 1)
ra, dec, _ = hr8799_sys.compute_all_orbits(params_arr, epochs)

# for example, to get the Dec of planet 1 at each epoch:
dec_planet1 = dec[:,1,:].flatten()

# to get the RA of the star at each epoch:
ra_star = ra[:,0,:].flatten()


def calc_diff():
    import orbitize.kepler
    import matplotlib.pyplot as plt
    
    rra, rde, rvz = calc_orbit(epochs, sma,ecc,inc,aop,pan,tau,plx,mtot,tau_ref_epoch, m_pl)
    ora = ra[:,1:5,0]
    odec = dec[:,1:5,0]
    

    delta_ra = abs(rra-ora)
    delta_de = abs(rde-odec)
    #delta_vz = abs(rvz-_)
    #yepochs = epoch/365.25

    if len(sma)==1:
        plt.plot(epochs, delta_ra, label = 'Planet X: RA offsets')
        plt.plot(epochs, delta_de, label = 'Planet X: Dec offsets')
        #plt.plot(epochs, delta_vz, label = 'Planet X: RV offsets')

    elif len(sma)==4:
        
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Massive Orbits in Rebound vs. Orbitize approx.')

        ax1.plot(epochs, delta_ra[:,0], 'brown', label = 'Planet E: RA offsets') #first planet
        ax2.plot(epochs, delta_de[:,0], 'red', label = 'Planet E: Dec offsets')
        #plt.plot(epochs, delta_vz[:,0], 'pink', label = 'Planet B: RV offsets')

        ax1.plot(epochs, delta_ra[:,1], 'coral', label = 'Planet D: RA offsets') #second planet
        ax2.plot(epochs, delta_de[:,1], 'orange', label = 'Planet D: Dec offsets')
        #plt.plot(epochs, delta_vz[:,1], 'gold', label = 'Planet C: RV offsets')

        ax1.plot(epochs, delta_ra[:,2], 'greenyellow', label = 'Planet C: RA offsets') #third planet
        ax2.plot(epochs, delta_de[:,2], 'green', label = 'Planet C: Dec offsets')
        #plt.plot(epochs, delta_vz[:,2], 'darkgreen', label = 'Planet D: RV offsets')        


        ax1.plot(epochs, delta_ra[:,3], 'dodgerblue', label = 'Planet B: RA offsets') #fourth planet
        ax2.plot(epochs, delta_de[:,3], 'blue', label = 'Planet B: Dec offsets')
        #plt.plot(epochs, delta_vz[:,3], 'indigo', label = 'Planet E: RV offsets')
        

    else:
        print('I dont feel like it')

    plt.xlabel('Epochs (Earth years)')
    plt.ylabel('milliarcseconds')
    ax1.legend()
    ax2.legend()
    plt.show()

def plot_orbit():

    rra, rdec, rvz = calc_orbit(epochs, sma,ecc,inc,aop,pan,tau,plx,mtot,tau_ref_epoch, m_pl)
    
    plt.plot(ra[:,1:5,0], dec[:,1:5,0], 'indigo', label = 'Orbitize approx.')
    plt.plot(ra[-1,1:5,0], dec[-1,1:5,0],'o')
    
    plt.plot(rra, rdec, 'r', label = 'Rebound', alpha = 0.25)
    plt.plot(rra[-1], rdec[-1], 'o', alpha = 0.25)
        
    plt.plot(0, 0, '*')
    plt.legend()
    plt.show()

calc_diff()
