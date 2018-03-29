""" 
This code should produce outputs detailed in `orbitize/kepler.py`
"""

import numpy as np
from orbitize import kepler

epochs = np.array([58159.4, 58161.1,58161.1]) # dates of observations
sma = np.array([3., 4., 2.6]) # semimajor axes
ecc = np.array([0.1,5.,.999]) # eccentricities
tau = np.array([0.5, 0., 1.]) # epochs of periastron passage
argp = np.array([0, np.pi, np.pi/4.]) # argument of periastron
lan = np.array([np.pi, np.pi/6., 2.*np.pi]) # longitude of ascending node
inc = np.array([0., np.pi, np.pi/6.]) # inclination angle
plx = np.array([3., 4.5, 3.6]) # parallax
mtot = np.array([1., 2.3, 1.2]) # total mass of system


# calling sequence
raoff, deoff, vz = kepler.calc_orbit(epochs, sma, ecc, tau, argp, lan, inc, plx, mtot, mass=0)

print("test code result:")
print(raoff, deoff, vz)