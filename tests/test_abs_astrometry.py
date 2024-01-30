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

    epochs = Time(np.array([0, 0.5, 1.0]) + 1991.25, format="decimalyear").mjd
    ra_model = np.array([0, 25, 0])
    dec_model = np.array([0, 25, 0])

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

    # just read in any file since we're not computing Hipparcos-related likelihoods.
    hip_num = "027321"
    num_secondary_bodies = 1
    path_to_iad_file = "{}H{}.d".format(DATADIR, hip_num)
    testHiPIAD = hipparcos.HipparcosLogProb(
        path_to_iad_file, hip_num, num_secondary_bodies
    )

    astrom_data = read_input.read_file(filename)
    mySystem = system.System(
        1, astrom_data, 1, 1, fit_secondary_mass=True, hipparcos_IAD=testHiPIAD
    )

    # Zero proper motion, but large parallax = yearly motion should only
    # reflect parallax. Check that this parallax-only model matches the data.
    plx = np.sqrt(25**2 + 25**2)
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
    plxonly_model = mySystem.compute_model(plx_only_params)
    assert False  # TODO

    # very high proper motion, but very small parallax = yearly motion should only
    # reflect proper motion
    plx = 0.0000001
    pm_ra = 25
    pm_dec = 25
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

    pmonly_expectation = np.array([[0, 0], [12.5, 12.5], [25.0, 25.0]])

    assert np.all(np.isclose(pmonly_model[0], pmonly_expectation, atol=1e-6))
    assert np.all(
        np.isclose(pmonly_model[1], np.zeros(pmonly_model[1].shape), atol=1e-6)
    )


if __name__ == "__main__":
    # test_1planet()
    test_arbitrary_abs_astrom()
