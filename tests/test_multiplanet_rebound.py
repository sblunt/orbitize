import os
import numpy as np
import astropy.table as table
import astropy
import astropy.units as u
import orbitize
import orbitize.read_input as read_input
import orbitize.kepler as kepler
import orbitize.system as system
import orbitize.basis as basis
import orbitize.priors as priors
import orbitize.driver as driver

### Skip this test on Windows since REBOUND doesn't work on Windows ###
import sys
import pytest
if sys.platform.startswith("win"):
    pytest.skip("Skipping REBOUND tests on Windows", allow_module_level=True)

try:
    import rebound
except ImportError:
     pytest.skip("Skipping REBOUND tests because REBOUND is not installed", allow_module_level=True)

def test_1planet():
    """
    Sanity check that things agree for 1 planet case
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

    sys = system.System(1, astrom_dat, mtot, plx, tau_ref_epoch=tau_ref_epoch)

    params = np.array([sma, ecc, inc, aop, pan, tau, plx, mtot])
    radec_orbitize, _ = sys.compute_model(params)
    ra_orb = radec_orbitize[:, 0]
    dec_orb = radec_orbitize[:, 1]


    # now project the orbit with rebound
    manom = basis.tau_to_manom(epochs[0], sma, mtot, tau, tau_ref_epoch)
    
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')


    # add star
    sim.add(m=mtot - mass_b)

    # add one planet
    sim.add(m=mass_b, a=sma, e=ecc, M=manom, omega=aop, Omega=pan+np.pi/2, inc=inc)
    ps = sim.particles

    sim.move_to_com()

    # Use Wisdom Holman integrator (fast), with the timestep being < 5% of inner planet's orbital period
    sim.integrator = "ias15"
    sim.dt = ps[1].P/1000.

    # integrate and measure star/planet separation 
    ra_reb = []
    dec_reb = []

    for t in epochs:
        sim.integrate(t/365.25)
        
        ra_reb.append(-(ps[1].x - ps[0].x)) # ra is negative x
        dec_reb.append(ps[1].y - ps[0].y)
        
    ra_reb = np.array(ra_reb)
    dec_reb = np.array(dec_reb)

    diff_ra = ra_reb - ra_orb/plx
    diff_dec = dec_reb - dec_orb/plx

    assert np.all(np.abs(diff_ra) < 1e-9)
    assert np.all(np.abs(diff_dec) < 1e-9)


def test_2planet_massive():
    """
    Compare multiplanet to rebound for planets with mass.
    """
    # generate a planet orbit
    mjup = u.Mjup.to(u.Msun)
    mass_b = 12 * mjup
    mass_c = 9 * mjup

    params = np.array([10, 0.1, np.radians(45), np.radians(45), np.radians(45), 0.5,
                       3, 0.1, np.radians(45), np.radians(190), np.radians(45), 0.2,
                       50, mass_b, mass_c, 1.5 - mass_b - mass_c])
    params_noc = np.array([10, 0.1, np.radians(45), np.radians(45), np.radians(45), 0.5,
                    3, 0.1, np.radians(45), np.radians(190), np.radians(45), 0.2,
                    50, mass_b, 0, 1.5 - mass_b])
    tau_ref_epoch = 0


    epochs = np.linspace(0, 365.25*10, 100) + tau_ref_epoch # nearly the full period, MJD

    # doesn't matter that this is right, just needs to be the same size. below doesn't include effect of c
    # just want to generate some measurements of plaent b to test compute model
    b_ra_model, b_dec_model, b_vz_model = kepler.calc_orbit(epochs, params[0], params[1], params[2], params[3], params[4], params[5], params[-2], params[-1], tau_ref_epoch=tau_ref_epoch)

    # generate some fake measurements of planet b, just to feed into system.py to test bookkeeping
    t = table.Table([epochs, np.ones(epochs.shape, dtype=int), b_ra_model, np.zeros(b_ra_model.shape), b_dec_model, np.zeros(b_dec_model.shape)], 
                     names=["epoch", "object" ,"raoff", "raoff_err","decoff","decoff_err"])
    filename = os.path.join(orbitize.DATADIR, "rebound_2planet_outer.csv")
    t.write(filename)

    #### TEST THE OUTER PLANET ####

    # create the orbitize system and generate model predictions using the ground truth
    astrom_dat = read_input.read_file(filename)

    sys = system.System(2, astrom_dat, params[-1], params[-4], tau_ref_epoch=tau_ref_epoch, fit_secondary_mass=True)

    # generate measurement
    radec_orbitize, _ = sys.compute_model(params)
    b_ra_orb = radec_orbitize[:, 0]
    b_dec_orb = radec_orbitize[:, 1]
    # debug, generate measurement without c having any mass
    radec_orb_noc, _ = sys.compute_model(params_noc)
    b_ra_orb_noc = radec_orb_noc[:,0]
    b_dec_orb_noc = radec_orb_noc[:,1]

    # check that planet c's perturbation is imprinted (nonzero))
    #assert np.all(b_ra_orb_noc != b_ra_orb)

    # now project the orbit with rebound
    b_manom = basis.tau_to_manom(epochs[0], params[0], params[-1]+params[-3], params[5], tau_ref_epoch)
    c_manom = basis.tau_to_manom(epochs[0], params[0+6], params[-1]+params[-2], params[5+6], tau_ref_epoch)
    
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')


    # add star
    sim.add(m=params[-1])

    # add two planets
    sim.add(m=mass_c, a=params[0+6], e=params[1+6], M=c_manom, omega=params[3+6], Omega=params[4+6]+np.pi/2, inc=params[2+6])
    sim.add(m=mass_b, a=params[0], e=params[1], M=b_manom, omega=params[3], Omega=params[4]+np.pi/2, inc=params[2])
    ps = sim.particles

    sim.move_to_com()

    # Use Wisdom Holman integrator (fast), with the timestep being < 5% of inner planet's orbital period
    sim.integrator = "ias15"
    sim.dt = ps[1].P/1000.

    # integrate and measure star/planet separation 
    b_ra_reb = []
    b_dec_reb = []

    for t in epochs:
        sim.integrate(t/365.25)
        
        b_ra_reb.append(-(ps[2].x - ps[0].x)) # ra is negative x
        b_dec_reb.append(ps[2].y - ps[0].y)
        
    b_ra_reb = np.array(b_ra_reb)
    b_dec_reb = np.array(b_dec_reb)

    diff_ra = b_ra_reb - b_ra_orb/params[6*2]
    diff_dec = b_dec_reb - b_dec_orb/params[6*2]

    # we placed the planets far apart to minimize secular interactions but there are still some, so relax precision
    assert np.all(np.abs(diff_ra)/(params[0]) < 1e-3)
    assert np.all(np.abs(diff_dec)/(params[0]) < 1e-3)

    ###### NOW TEST THE INNER PLANET #######

    # generate some fake measurements of planet c, just to feed into system.py to test bookkeeping
    t = table.Table([epochs, np.ones(epochs.shape, dtype=int)*2, b_ra_model, np.zeros(b_ra_model.shape), b_dec_model, np.zeros(b_dec_model.shape)], 
                     names=["epoch", "object" ,"raoff", "raoff_err","decoff","decoff_err"])
    filename = os.path.join(orbitize.DATADIR, "rebound_2planet_inner.csv")
    t.write(filename)

    # create the orbitize system and generate model predictions using the ground truth
    astrom_dat = read_input.read_file(filename)

    sys = system.System(2, astrom_dat, params[-1], params[-2], tau_ref_epoch=tau_ref_epoch, fit_secondary_mass=True)

    # generate measurement
    radec_orbitize, _ = sys.compute_model(params)
    c_ra_orb = radec_orbitize[:, 0]
    c_dec_orb = radec_orbitize[:, 1]
    
    # start the REBOUND sim again
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    # add star
    sim.add(m=params[-1])

    # add two planets
    sim.add(m=mass_c, a=params[0+6], e=params[1+6], M=c_manom, omega=params[3+6], Omega=params[4+6]+np.pi/2, inc=params[2+6])
    sim.add(m=mass_b, a=params[0], e=params[1], M=b_manom, omega=params[3], Omega=params[4]+np.pi/2, inc=params[2])
    ps = sim.particles

    sim.move_to_com()

    # Use Wisdom Holman integrator (fast), with the timestep being < 5% of inner planet's orbital period
    sim.integrator = "ias15"
    sim.dt = ps[1].P/1000.

    # integrate and measure star/planet separation 
    c_ra_reb = []
    c_dec_reb = []

    for t in epochs:
        sim.integrate(t/365.25)
        
        c_ra_reb.append(-(ps[1].x - ps[0].x)) # ra is negative x
        c_dec_reb.append(ps[1].y - ps[0].y)
        
    c_ra_reb = np.array(c_ra_reb)
    c_dec_reb = np.array(c_dec_reb)

    diff_ra = c_ra_reb - c_ra_orb/params[6*2]
    diff_dec = c_dec_reb - c_dec_orb/params[6*2]

    # planet is 3 times closer, so roughly 3 times larger secular errors. 
    assert np.all(np.abs(diff_ra)/(params[0]) < 3e-3)
    assert np.all(np.abs(diff_dec)/(params[0]) < 3e-3)


def test_2planet_nomass():
    """
    Compare multiplanet to rebound for planets with mass.
    """
    # generate a planet orbit
    mjup = u.Mjup.to(u.Msun)
    mass_b = 0 * mjup
    mass_c = 0 * mjup

    params = np.array([10, 0.1, np.radians(45), np.radians(45), np.radians(45), 0.5,
                       3, 0.1, np.radians(45), np.radians(190), np.radians(45), 0.2,
                       1, mass_b, mass_c, 1.5 - mass_b - mass_c])
    tau_ref_epoch = 0


    epochs = np.linspace(0, 365.25*4, 100) + tau_ref_epoch # nearly the full period, MJD

    # doesn't matter that this is right, just needs to be the same size. below doesn't include effect of c
    # just want to generate some measurements of plaent b to test compute model
    b_ra_model, b_dec_model, b_vz_model = kepler.calc_orbit(epochs, params[0], params[1], params[2], params[3], params[4], params[5], params[-2], params[-1], tau_ref_epoch=tau_ref_epoch)

    # generate some fake measurements of planet b, just to feed into system.py to test bookkeeping
    t = table.Table([epochs, np.ones(epochs.shape, dtype=int), b_ra_model, np.zeros(b_ra_model.shape), b_dec_model, np.zeros(b_dec_model.shape)], 
                     names=["epoch", "object" ,"raoff", "raoff_err","decoff","decoff_err"])
    filename = os.path.join(orbitize.DATADIR, "rebound_2planet.csv")
    t.write(filename)

    # create the orbitize system and generate model predictions using the ground truth
    astrom_dat = read_input.read_file(filename)

    sys = system.System(2, astrom_dat, params[-1], params[-2], tau_ref_epoch=tau_ref_epoch, fit_secondary_mass=True)

    # generate measurement
    radec_orbitize, _ = sys.compute_model(params)
    b_ra_orb = radec_orbitize[:, 0]
    b_dec_orb = radec_orbitize[:, 1]

    # now project the orbit with rebound
    b_manom = basis.tau_to_manom(epochs[0], params[0], params[-1]+params[-3], params[5], tau_ref_epoch)
    c_manom = basis.tau_to_manom(epochs[0], params[0+6], params[-1]+params[-2], params[5+6], tau_ref_epoch)
    
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    # add star
    sim.add(m=params[-1])

    # add two planets
    sim.add(m=mass_c, a=params[0+6], e=params[1+6], M=c_manom, omega=params[3+6], Omega=params[4+6]+np.pi/2, inc=params[2+6])
    sim.add(m=mass_b, a=params[0], e=params[1], M=b_manom, omega=params[3], Omega=params[4]+np.pi/2, inc=params[2])
    ps = sim.particles

    sim.move_to_com()

    # Use Wisdom Holman integrator (fast), with the timestep being < 5% of inner planet's orbital period
    sim.integrator = "ias15"
    sim.dt = ps[1].P/1000.

    # integrate and measure star/planet separation 
    b_ra_reb = []
    b_dec_reb = []

    for t in epochs:
        sim.integrate(t/365.25)
        
        b_ra_reb.append(-(ps[2].x - ps[0].x)) # ra is negative x
        b_dec_reb.append(ps[2].y - ps[0].y)
        
    b_ra_reb = np.array(b_ra_reb)
    b_dec_reb = np.array(b_dec_reb)

    diff_ra = b_ra_reb - b_ra_orb/params[6*2]
    diff_dec = b_dec_reb - b_dec_orb/params[6*2]

    # should be as good as the one planet case
    assert np.all(np.abs(diff_ra)/(params[0]*params[6*2]) < 1e-9)
    assert np.all(np.abs(diff_dec)/(params[0]*params[6*2]) < 1e-9)


if __name__ == "__main__":
    #test_1planet()
    test_2planet_massive()
    #test_2planet_nomass()
    #test_OFTI_multiplanet()




