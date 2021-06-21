import numpy as np
import os
import astropy.table as table
import astropy.units as u

import orbitize
from orbitize import kepler, read_input, system

def test_1planet():
    """
    Check that for the 2-body case, the stellar orbit looks as expected
    """

    # generate a planet orbit
    sma = 1
    ecc = 0.1
    inc = np.radians(45)
    aop = np.radians(45)
    pan = np.radians(45)
    tau = 0.5
    plx = 1
    mtot = 1
    tau_ref_epoch = 0
    mjup = u.Mjup.to(u.Msun)
    mass_b = 12 * mjup

    epochs = np.linspace(0, 300, 100) + tau_ref_epoch # nearly the full period, MJD

    ra_model, dec_model, vz_model = kepler.calc_orbit(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, tau_ref_epoch=tau_ref_epoch)

    # generate some fake measurements just to feed into system.py to test bookkeeping
    t = table.Table([epochs, np.ones(epochs.shape, dtype=int), ra_model, np.zeros(ra_model.shape), dec_model, np.zeros(dec_model.shape)], 
                     names=["epoch", "object" ,"raoff", "raoff_err","decoff","decoff_err"])
    filename = os.path.join(orbitize.DATADIR, "rebound_1planet.csv")
    t.write(filename)

    # create the orbitize system and generate model predictions using the ground truth
    astrom_dat = read_input.read_file(filename)

    sys = system.System(1, astrom_dat, mtot, plx, tau_ref_epoch=tau_ref_epoch, fit_secondary_mass=True)
    sys.track_planet_perturbs = True

    params = np.array([sma, ecc, inc, aop, pan, tau, plx, mass_b, mtot])
    ra, dec, _ = sys.compute_all_orbits(params)

    # the planet and stellar orbit should just be scaled versions of one another
    planet_ra = ra[:,1,:]
    planet_dec = dec[:,1,:]
    star_ra = ra[:,0,:]
    star_dec = dec[:,0,:]
    assert np.all(star_ra + (mass_b / mtot) * planet_ra < 1e-3)

if __name__ == '__main__':
    test_1planet()