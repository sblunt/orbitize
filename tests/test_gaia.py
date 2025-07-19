import numpy as np
import os
from orbitize import DATADIR
from orbitize import hipparcos, gaia, basis, system, read_input, sampler, results


def test_dr2_edr3():
    """
    Test that both DR2 and eDR3 retrieval gives ballpark similar values for
    beta Pic
    """
    hip_num = "027321"  # beta Pic
    edr3_num = 4792774797545800832
    dr2_number = 4792774797545105664

    num_secondary_bodies = 1
    path_to_iad_file = "{}HIP{}.d".format(DATADIR, hip_num)

    myHip = hipparcos.HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)

    dr3Gaia = gaia.GaiaLogProb(edr3_num, myHip, dr="edr3")
    dr2Gaia = gaia.GaiaLogProb(dr2_number, myHip, dr="dr2")

    assert np.isclose(dr2Gaia.ra, dr3Gaia.ra, atol=0.1)  # abs tolerance in degrees


def test_system_setup():
    """
    Test that a System object with Hipparcos and Gaia is initialized correctly
    """
    hip_num = "027321"  # beta Pic
    edr3_num = 4792774797545800832
    num_secondary_bodies = 1
    path_to_iad_file = "{}HIP{}.d".format(DATADIR, hip_num)

    myHip = hipparcos.HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)
    myGaia = gaia.GaiaLogProb(edr3_num, myHip, dr="edr3")

    input_file = os.path.join(DATADIR, "betaPic.csv")
    plx = 51.5

    num_secondary_bodies = 1
    data_table = read_input.read_file(input_file)

    betaPic_system = system.System(
        num_secondary_bodies,
        data_table,
        1.75,
        plx,
        hipparcos_IAD=myHip,
        gaia=myGaia,
        fit_secondary_mass=True,
        mass_err=0.01,
        plx_err=0.01,
    )

    assert betaPic_system.labels == [
        "sma1",
        "ecc1",
        "inc1",
        "aop1",
        "pan1",
        "tau1",
        "plx",
        "pm_ra",
        "pm_dec",
        "alpha0",
        "delta0",
        "m1",
        "m0",
    ]

    assert betaPic_system.fit_secondary_mass
    assert betaPic_system.track_planet_perturbs


def test_valueerror():
    """
    Check that if I don't say dr2 or edr3, I get a value error
    """
    hip_num = "027321"  # beta Pic
    edr3_num = 4792774797545800832
    num_secondary_bodies = 1
    path_to_iad_file = "{}HIP{}.d".format(DATADIR, hip_num)

    myHip = hipparcos.HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)
    try:
        myGaia = gaia.GaiaLogProb(edr3_num, myHip, dr="dr3")
        assert False, "Test failed!"
    except ValueError:
        pass


def test_orbit_calculation():
    """
    Test that the Gaia module correctly calculates log likelihood
    for simulated astrometric motion due to:
        1) proper motion only
        2) fitted offset in Hipparcos positon only
        3) orbital motion only

    NOTE: this only works as long as lnlike is defined as (data - model) in
        the Gaia module (i.e. no constant offset term applied), since
        I'm checking that the absolute likelihood probability is 1 when data =
        model.
    """

    sma = 1
    ecc = 0
    inc = 0
    aop = 0
    pan = 0
    tau = 0

    pm_a = 0
    pm_d = 0

    plx = 100  # [mas]
    m0 = 1
    m1 = 1
    a0 = 0
    d0 = 0

    hip_num = "027321"  # beta Pic
    edr3_num = 4792774797545800832
    num_secondary_bodies = 1
    path_to_iad_file = "{}HIP{}.d".format(DATADIR, hip_num)

    myHip = hipparcos.HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)
    myGaia = gaia.GaiaLogProb(edr3_num, myHip, dr="edr3")

    param_idx = {
        "sma1": 0,
        "ecc1": 1,
        "inc1": 2,
        "aop1": 3,
        "pan1": 4,
        "tau1": 5,
        "plx": 6,
        "m0": 7,
        "m1": 8,
        "alpha0": 9,
        "delta0": 10,
        "pm_ra": 11,
        "pm_dec": 12,
    }

    # Case 1: only proper motion explains Gaia-Hip offset
    pm_a = 100  # [mas/yr]
    pm_d = 100  # [mas/yr]
    sma = 1e-17

    raoff = np.zeros((2, 1))
    deoff = np.zeros((2, 1))

    myGaia.ra = myHip.alpha0 + (
        myGaia.mas2deg
        * pm_a
        * (myGaia.gaia_epoch.jyear - myGaia.hipparcos_epoch.jyear)
        / np.cos(np.radians(myHip.delta0))
    )
    myGaia.dec = myHip.delta0 + (
        myGaia.mas2deg * pm_d * (myGaia.gaia_epoch.jyear - myGaia.hipparcos_epoch.jyear)
    )
    test_samples = [sma, ecc, inc, aop, pan, tau, plx, m0, m1, a0, d0, pm_a, pm_d]

    lnlike = myGaia.compute_lnlike(raoff, deoff, test_samples, param_idx)

    assert np.isclose(np.exp(lnlike), 1)

    # Case 2: only H0 offset explains Gaia-Hip offset
    test_samples[param_idx["pm_dec"]] = 0
    test_samples[param_idx["pm_ra"]] = 0
    a0 = 100
    d0 = 100
    test_samples[param_idx["alpha0"]] = a0  # [mas]
    test_samples[param_idx["delta0"]] = d0  # [mas]

    myGaia.ra = myHip.alpha0 + myGaia.mas2deg * a0 / np.cos(np.radians(myHip.delta0))
    myGaia.dec = myHip.delta0 + myGaia.mas2deg * d0

    lnlike = myGaia.compute_lnlike(raoff, deoff, test_samples, param_idx)

    assert np.isclose(np.exp(lnlike), 1)

    # Case 3: only orbital motion explains Gaia-Hip offset
    test_samples[param_idx["alpha0"]] = 0
    test_samples[param_idx["delta0"]] = 0

    mas2arcsec = 1e-3
    deg2arcsec = 3600

    myGaia.ra = myHip.alpha0
    myGaia.dec = myHip.delta0 + 1

    sma = 2 * (myGaia.dec - myHip.delta0) * deg2arcsec * (plx * mas2arcsec)  # [au]
    per = 2 * (myGaia.gaia_epoch.jyear - myGaia.hipparcos_epoch.jyear)  # [yr]
    mtot = sma**3 / per**2

    test_samples[param_idx["sma1"]] = sma
    test_samples[param_idx["m0"]] = mtot / 2
    test_samples[param_idx["m1"]] = mtot / 2

    # passes through peri (+sma decl for e=0 orbits) at Hipparcos epoch
    # -> @ Gaia epoch, primary should be at +sma decl
    tau = basis.tp_to_tau(myGaia.hipparcos_epoch.jyear, 58849, per)
    test_samples[param_idx["tau1"]] = tau

    # choose sma and mass so that Hipparcos/Gaia difference is only due to orbit.
    deoff[1, :] = (myGaia.dec - myHip.delta0) / myGaia.mas2deg
    deoff[0, :] = 0
    lnlike = myGaia.compute_lnlike(raoff, deoff, test_samples, param_idx)

    assert np.isclose(np.exp(lnlike), 1)


def test_hgca():
    """
    Tests that the HGCA module works for beta Pic b.

    Checks can download HGCA catalog if not available, and checks GRAVITY Collaboration et al. 2020
    orbital parameters against the data
    """
    iad_filepath = os.path.join(DATADIR, "HIP027321.d")
    gost_filepath = os.path.join(DATADIR, "gaia_edr3_betpic_epochs.csv")
    astrometry_filepath = os.path.join(DATADIR, "betaPic.csv")

    hipparcos_lnprob = hipparcos.HipparcosLogProb(iad_filepath, 27321, 1)
    hgca_lnprob = gaia.HGCALogProb(27321, hipparcos_lnprob, gost_filepath)

    # test a few things were read in correctly
    assert np.all(np.isfinite(hgca_lnprob.hip_pm))
    assert len(hgca_lnprob.hipparcos_epoch) > 0
    assert len(hgca_lnprob.gaia_epoch) > 0

    # Initialize System object which stores data & sets priors
    data_table = read_input.read_file(astrometry_filepath)
    this_system = system.System(
        1,
        data_table,
        1.75,
        51.44,
        mass_err=0.05,
        plx_err=0.12,
        tau_ref_epoch=55000,
        fit_secondary_mass=True,
        gaia=hgca_lnprob,
    )

    # create sampler just for lnlike function
    this_sampler = sampler.MCMC(this_system, 1, 100, 1)

    # params from GRAVITY Collaboration et al. 2020 fit to HGCA DR2
    params = np.array(
        [
            [
                1.04e01,
                1.44e-01,
                1.5534e00,
                3.5325e00,
                5.5670e-01,
                1.80968e-01,
                5.1456e01,
                1.367e-02,
                1.783,
            ]
        ]
    )

    chi2 = this_sampler._logl(params)

    assert np.isfinite(chi2)

    # briefly run sampler and save result
    this_sampler.run_sampler(100, 0)
    this_sampler.results.save_results("hgca_test.hdf5")

    # check that the data stays the same
    res = results.Results()
    res.load_results("hgca_test.hdf5")
    new_hgca_lnprob = res.system.gaia
    assert isinstance(new_hgca_lnprob, gaia.HGCALogProb)
    assert new_hgca_lnprob.hip_hg_dpm_radec_corr == hgca_lnprob.hip_hg_dpm_radec_corr

    os.remove("hgca_test.hdf5")


def test_nointernet():
    """
    Test that the internet-less object setup works
    """
    hip_num = "027321"  # beta Pic
    dr2_number = 4792774797545105664

    num_secondary_bodies = 1
    path_to_iad_file = "{}HIP{}.d".format(DATADIR, hip_num)

    myHip = hipparcos.HipparcosLogProb(path_to_iad_file, hip_num, num_secondary_bodies)

    _ = gaia.GaiaLogProb(
        dr2_number,
        myHip,
        dr="dr2",
        query=False,
        gaia_data={"ra": 0, "dec": 0, "ra_error": 0, "dec_error": 0},
    )


if __name__ == "__main__":
    test_nointernet()
    test_dr2_edr3()
    test_system_setup()
    test_valueerror()
    test_orbit_calculation()
    test_hgca()
