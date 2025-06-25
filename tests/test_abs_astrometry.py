import numpy as np
import os
import astropy.table as table
from astropy.time import Time
import astropy.units as u

import orbitize
from orbitize import kepler, read_input, system, hipparcos, DATADIR


def test_1planet():
    """
    Check that for the 2-body case, the primary orbit around the barycenter
    is equal to -m2/(m1 + m2) times the secondary orbit around the primary.
    """

    # generate a planet orbit
    sma = 1
    ecc = 0.1
    inc = np.radians(45)
    aop = np.radians(45)
    pan = np.radians(45)
    tau = 0.5
    plx = 1
    m0 = 1
    tau_ref_epoch = 0
    mjup = u.Mjup.to(u.Msun)
    mass_b = 100 * mjup
    mtot = mass_b + m0

    epochs = np.linspace(0, 300, 100) + tau_ref_epoch  # nearly the full period, MJD

    ra_model, dec_model, _ = kepler.calc_orbit(
        epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, tau_ref_epoch=tau_ref_epoch
    )

    # generate some fake measurements to feed into system.py to test bookkeeping
    t = table.Table(
        [
            epochs,
            np.ones(epochs.shape, dtype=int),
            ra_model,
            np.zeros(ra_model.shape),
            dec_model,
            np.zeros(dec_model.shape),
        ],
        names=["epoch", "object", "raoff", "raoff_err", "decoff", "decoff_err"],
    )
    filename = os.path.join(orbitize.DATADIR, "rebound_1planet.csv")
    t.write(filename, overwrite=True)

    # create the orbitize system and generate model predictions using ground truth
    astrom_dat = read_input.read_file(filename)

    sys = system.System(
        1, astrom_dat, m0, plx, tau_ref_epoch=tau_ref_epoch, fit_secondary_mass=True
    )
    sys.track_planet_perturbs = True

    params = np.array([sma, ecc, inc, aop, pan, tau, plx, mass_b, m0])
    ra, dec, _ = sys.compute_all_orbits(params)

    # the planet and stellar orbit should just be scaled versions of one another
    planet_ra = ra[:, 1, :]
    planet_dec = dec[:, 1, :]
    star_ra = ra[:, 0, :]
    star_dec = dec[:, 0, :]

    assert np.all(np.abs(star_ra + (mass_b / mtot) * planet_ra) < 1e-16)
    assert np.all(np.abs(star_dec + (mass_b / mtot) * planet_dec) < 1e-16)

    # remove the created csv file to clean up
    os.system("rm {}".format(filename))


def test_arbitrary_abs_astrom():
    """
    Test that proper motion and parallax model parameters are applied correctly
    when we have astrometry from an arbitrary (i.e. not Hipparcos or Gaia)
    instrument. This test assumes test particle (i.e. zero-mass) companions.
    """

    # just read in any file since we're not computing Hipparcos-related likelihoods.
    hip_num = "027321"
    num_secondary_bodies = 1
    path_to_iad_file = "{}H{}.d".format(DATADIR, hip_num)
    testHiPIAD = hipparcos.HipparcosLogProb(
        path_to_iad_file, hip_num, num_secondary_bodies
    )

    epochs_astropy = Time(
        np.array([0, 0.5, 1.0]) + testHiPIAD.alphadec0_epoch, format="decimalyear"
    )
    epochs = epochs_astropy.mjd
    ra_model = np.zeros(epochs.shape)
    dec_model = np.zeros(epochs.shape)

    # generate some fake measurements to feed into system.py to test bookkeeping
    t = table.Table(
        [
            epochs,
            np.zeros(epochs.shape, dtype=int),
            ra_model,
            np.zeros(epochs.shape),
            dec_model,
            np.zeros(epochs.shape),
        ],
        names=["epoch", "object", "raoff", "raoff_err", "decoff", "decoff_err"],
    )
    filename = os.path.join(orbitize.DATADIR, "rebound_1planet.csv")
    t.write(filename, overwrite=True)

    astrom_data = read_input.read_file(filename)
    mySystem = system.System(
        1, astrom_data, 1, 1, fit_secondary_mass=True, hipparcos_IAD=testHiPIAD
    )

    # Test case 1: zero proper motion, but large parallax = yearly motion should only
    # reflect parallax
    plx = 100
    pm_ra = 0
    pm_dec = 0
    alpha0 = 0
    delta0 = 0
    m1 = 1
    m0 = 1e-10

    plx_only_params = np.array(
        [
            1,  # start test particle params
            0,
            0,
            0,
            0,
            0,  # end test particle params
            plx,
            pm_ra,
            pm_dec,
            alpha0,
            delta0,
            m1,
            m0,
        ]
    )

    plxonly_fullorbit_ra, plxonly_fullorbit_dec, _ = mySystem.compute_all_orbits(
        plx_only_params, epochs=np.linspace(epochs[0], epochs[0] + 365.25 / 2, int(1e6))
    )

    # check that min and max of RA and Dec outputs are close to 0 and plx magnitude,
    # respectively
    assert np.isclose(0, np.min(np.abs(plxonly_fullorbit_ra)), atol=1e-4)
    assert np.isclose(0, np.min(np.abs(plxonly_fullorbit_dec)), atol=1e-4)
    assert np.isclose(-100, np.min(plxonly_fullorbit_ra), atol=1e-4)
    assert np.isclose(100, np.max(plxonly_fullorbit_dec), atol=1e-4)

    # Test case 2: very high proper motion, but very small parallax = motion
    # should only reflect proper motion
    plx = 1e-10
    pm_ra = 100
    pm_dec = 100
    pm_only_params = np.array(
        [
            1,  # start test particle params
            0,
            0,
            0,
            0,
            0,  # end test particle params
            plx,
            pm_ra,
            pm_dec,
            alpha0,
            delta0,
            m1,
            m0,
        ]
    )

    pmonly_model = mySystem.compute_model(pm_only_params)

    cosdelta0 = np.cos(np.radians(mySystem.pm_plx_predictor.delta0))

    pmonly_expectation = np.array(
        [[0, 0], [50 / cosdelta0, 50], [100.0 / cosdelta0, 100.0]]
    )

    assert np.all(np.isclose(pmonly_model[0], pmonly_expectation))
    assert np.all(np.isclose(pmonly_model[1], np.zeros(pmonly_model[1].shape)))


if __name__ == "__main__":
    test_1planet()
    test_arbitrary_abs_astrom()
