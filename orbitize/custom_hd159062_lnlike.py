import orbitize.kepler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time

"""
This chi2 log likelihood contains the functions to calculate the
instantaneous velocity of a star, and the difference in velocities over a span
of time. The goal of this chi2 is to place
further constraints on the system.
"""

def vel_star(theta,time):
    #theta are the parameters (array or list form), and time are the observation epochs (also a list or an array)
    """Arguments (theta):
        [0]: sma (semi-major axis)
        [1]: ecc (eccentricity between 0.0 and 1.0)
        [2]: inc (inclination)
        [3]: argp (argument of periastron)
        [4]: lan (logitude of the ascending node)
        [5]: tau (time since periastron as a fraction of the orbit)
        [6]: plx (parallax in mas)
        [7]: m1 (companion mass in solar masses)
        [8]: m0 (stellar mass in solar masses)

        time has shape len(N_obs)

        Returns: array of length(times) of the instantaneous velocities of the
        star."""

    m0 = theta[-1]
    m1 = theta[-2]
    sma = theta[0]
    ecc = theta[1]
    argp = theta[3]
    lan = theta[4]
    inc = theta[2]
    tau = theta[5]
    plx = theta[6]

    #first thing we have to do is figure out the average angular velocity in rad/day:
    P = (np.sqrt(sma**3/(m0 + m1)))*365.25 #returning period in days
    n = 2*np.pi/P

    #time of periastron in days:
    tperi = P*tau

    #Mean anomaly as a function of time in radians for both times:
    M = n*(time - tperi)
    M = np.mod(M,2*np.pi)

    ecc_arr = np.ones(len(M))*ecc

    #eccentric anomalies:
    E = orbitize.kepler._calc_ecc_anom(M,ecc_arr)

    nu = 2*np.arctan(np.sqrt((1 + ecc)/(1 - ecc))*np.tan(E/2))

    u = argp + nu

    #getting semi-major axis in AU from the period:

    #dx/dt,dy/dt are arrays of length time
    r = 2*np.pi*sma/(P*np.sqrt(1 - ecc**2))

    dxdt = r*(np.cos(inc)*np.cos(lan)*(ecc*np.cos(argp) + np.cos(u)) - \
              np.sin(lan)*(ecc*np.sin(argp) + np.sin(u)))
    dydt = -r*(np.cos(inc)*np.sin(lan)*(ecc*np.cos(argp) + np.cos(u)) + \
              np.cos(lan)*(ecc*np.sin(argp) + np.sin(u)))

    #getting velocity of star:

    dxdt_s = dxdt*-(m0/m1)
    dydt_s = dydt*-(m0/m1)

    #getting dra/dt and ddec/dt in arcseconds per year:
    dradt = dxdt*plx
    ddecdt = dydt*plx

    return dradt,ddecdt

def dvel(theta):
    epochs = np.array([Time(1991.25,format='mjd').jd,
        Time(2015.50,format='mjd').jd])
    vels = vel_star(theta,epochs)
    dv = np.subtract(vels[0],vels[1])
    return dv

def custom_chi2_loglike(theta):
    #Hipparcos pm:
    pm_ra_H = 174.31 #mas/yr
    e_pm_ra_H = 0.66 #mas/yr
    pm_dec_H = 76.82
    e_pm_dec_H = 0.63

    #Gaia pm:
    pm_ra_G = 169.747
    e_pm_ra_G = 0.061
    pm_dec_G = 77.164
    e_pm_dec_G = 0.058

    #observational velocities:
    obs_dradt = pm_ra_G - pm_ra_H
    obs_ddecdt = pm_dec_G - pm_dec_H
    obs_dvel = np.array([obs_dradt,obs_ddecdt])

    #errors:
    obs_err_ra = np.sqrt(e_pm_ra_G**2 + e_pm_ra_H**2)
    obs_err_dec = np.sqrt(e_pm_dec_G**2 + e_pm_dec_H**2)

    obs_err = np.array([obs_err_ra,obs_err_dec])

    #model:

    mod_dvel = dvel(theta)

    #residual: data - model
    residual = obs_dvel - mod_dvel

    #chi2 and likelihood functions:
    chi2 = -0.5 * (residual**2 / obs_err**2) - np.log(np.sqrt(2.0*np.pi*obs_err**2))
    return chi2
