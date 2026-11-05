"""
Tests for Keplerian + acceleration fits,
``System(acceleration_and_keplerian_fit=True)``: a Keplerian orbit plus a
constant acceleration of the star (``acc_ra``, ``acc_dec``), e.g. from an
outer companion that isn't modelled. Gaia DR4 epoch astrometry is required;
relative astrometry and Hipparcos IAD can be added.
"""

from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from orbitize import DATADIR, gaia, hipparcos, plot, priors, read_input, sampler, system

REF_EPOCH_MJD = 57936.375  # J2017.5, the DR4 reference epoch [MJD]

# a 3-yr orbit (fits inside the 5-yr DR4 window) plus an acceleration
TRUTH = {
    "sma1": 1.5,
    "ecc1": 0.2,
    "inc1": np.radians(60.0),
    "aop1": np.radians(40.0),
    "pan1": np.radians(110.0),
    "tau1": 0.3,
    "plx": 20.0,
    "pm_ra": 5.0,
    "pm_dec": -12.0,
    "alpha0": 0.1,
    "delta0": -0.1,
    "acc_ra": 0.05,
    "acc_dec": -0.04,
    "m1": 0.01,
    "m0": 1.0,
}

HIP_DATA = {
    "RArad": 86.82118072,
    "e_RArad": 0.1,
    "DErad": -51.06671341,
    "e_DErad": 0.11,
    "Plx": 51.44,
    "e_Plx": 0.12,
    "pmRA": 4.65,
    "e_pmRA": 0.11,
    "pmDE": 83.1,
    "e_pmDE": 0.15,
    "F2": -1.63,
    "Sn": 5,
    "var": 0.0,
}


def mock_dr4(n_obs=40, err=0.05, seed=0):
    """Mock Gaia DR4 along-scan data over 5 yr centred on J2017.5."""
    rng = np.random.default_rng(seed)
    epochs = REF_EPOCH_MJD + np.linspace(-2.5, 2.5, n_obs) * 365.25
    data = Table(
        {
            "obs_time_tcb": epochs,
            "centroid_pos_al": np.zeros(n_obs),
            "centroid_pos_error_al": np.full(n_obs, err),
            "scan_pos_angle": rng.uniform(0, 2 * np.pi, n_obs),
            "parallax_factor_al": np.sin(2 * np.pi * (epochs - REF_EPOCH_MJD) / 365.25),
            "field_of_view": np.zeros(n_obs, dtype=int),
        }
    )
    data.meta.update(time_format="mjd", time_scale="tcb")
    return gaia.DR4LogProb(0, data, ref_epoch_mjd=REF_EPOCH_MJD)


def radec_table(tmp_path, obj, years=(2015.0, 2017.0, 2018.5, 2020.0), err=0.5):
    """Write ra/dec astrometry of ``obj`` at ``years`` to a csv file and read it
    back with ``read_input.read_file``. For obj=1 the values are a rough
    companion offset [mas]; for obj=0 (the star) they are zero."""
    epochs = Time(np.array(years), format="jyear").mjd
    n = len(epochs)
    value = 50.0 if obj else 0.0
    t = Table(
        [epochs, np.full(n, obj), np.full(n, value), np.full(n, err), np.full(n, value), np.full(n, err)],
        names=["epoch", "object", "raoff", "raoff_err", "decoff", "decoff_err"],
    )
    filename = tmp_path / "astrometry_{}.csv".format(obj)
    t.write(filename, overwrite=True)
    return read_input.read_file(str(filename))


def hip_iad():
    return hipparcos.HipparcosLogProb(
        "{}HIP027321.d".format(DATADIR), "027321", 1, hip_data=HIP_DATA
    )


def kep_system(dr4, data_table=None, acceleration=True, **kwargs):
    """One-companion System fitting the companion mass, with or without the
    acceleration terms."""
    return system.System(
        num_secondary_bodies=1,
        data_table=data_table,
        stellar_or_system_mass=1.0,
        plx=TRUTH["plx"],
        mass_err=0.1,
        plx_err=0.5,
        fit_secondary_mass=True,
        gaia=dr4,
        acceleration_and_keplerian_fit=acceleration,
        **kwargs,
    )


def params_from(sys, values):
    return np.array([values[label] for label in sys.labels], dtype=float)


def lnlike(sys, params):
    return sampler.MCMC(sys, num_temps=1, num_walkers=32, num_threads=1)._logl(params)


def test_labels():
    """The accelerations come after the DR4 astrometric parameters and
    before the masses."""
    sys = kep_system(mock_dr4())

    assert sys.labels == [
        "sma1", "ecc1", "inc1", "aop1", "pan1", "tau1",
        "plx", "pm_ra", "pm_dec", "alpha0", "delta0",
        "acc_ra", "acc_dec", "m1", "m0",
    ]
    assert sys.acceleration_and_keplerian_fit
    assert not sys.acceleration_only


def test_needs_dr4(tmp_path):
    """The acceleration is measured from the DR4 reference epoch, so the fit
    requires a DR4LogProb."""
    with pytest.raises(ValueError, match="Gaia DR4"):
        kep_system(None, data_table=radec_table(tmp_path, obj=1))


def test_zero_acceleration_matches_keplerian_fit():
    """With acc_ra = acc_dec = 0 the likelihood is the same as a plain
    Keplerian DR4 fit with the same parameters."""
    dr4 = mock_dr4()
    with_acc = kep_system(dr4)
    without_acc = kep_system(dr4, acceleration=False)

    values = dict(TRUTH, acc_ra=0.0, acc_dec=0.0)
    np.testing.assert_allclose(
        lnlike(with_acc, params_from(with_acc, values)),
        lnlike(without_acc, params_from(without_acc, values)),
    )


def test_acceleration_adds_to_star_only():
    """compute_all_orbits adds -acc*dt^2 (dt from the DR4 reference epoch) to
    the star's offsets and leaves the companion's unchanged."""
    dr4 = mock_dr4()
    sys = kep_system(dr4)
    epochs = dr4.epochs_mjd
    params = params_from(sys, TRUTH)
    no_acc = params_from(sys, dict(TRUTH, acc_ra=0.0, acc_dec=0.0))

    ra, dec, _, _ = sys.compute_all_orbits(params, epochs=epochs)
    ra0, dec0, _, _ = sys.compute_all_orbits(no_acc, epochs=epochs)

    dt = (epochs - REF_EPOCH_MJD) / 365.25
    np.testing.assert_allclose(ra[:, 0, 0] - ra0[:, 0, 0], -0.5 * TRUTH["acc_ra"] * dt**2)
    np.testing.assert_allclose(dec[:, 0, 0] - dec0[:, 0, 0], -0.5 * TRUTH["acc_dec"] * dt**2)
    np.testing.assert_array_equal(ra[:, 1, 0], ra0[:, 1, 0])
    np.testing.assert_array_equal(dec[:, 1, 0], dec0[:, 1, 0])


def test_acceleration_in_along_scan_model():
    """In the AL model the acceleration enters as +acc*dt^2 projected on the
    scan direction, the same sign as the proper motion."""
    dr4 = mock_dr4()
    sys = kep_system(dr4)
    params = params_from(sys, TRUTH)
    no_acc = params_from(sys, dict(TRUTH, acc_ra=0.0, acc_dec=0.0))

    diff = plot._model_along_scan_dr4(params, sys, dr4) - plot._model_along_scan_dr4(
        no_acc, sys, dr4
    )
    expected = 0.5 * dr4.dt_yr**2 * (
        TRUTH["acc_ra"] * dr4.sin_scan + TRUTH["acc_dec"] * dr4.cos_scan
    )
    np.testing.assert_allclose(diff, expected, rtol=0, atol=1e-10)


def test_fitted_acceleration_recovers_injected():
    """Inject an acceleration into data made from the Keplerian model (as in
    the acceleration_only_fit notebook), then fit it by least squares with
    the orbit held fixed: the injected values come back exactly."""
    dr4 = mock_dr4()
    sys = kep_system(dr4)
    no_acc = params_from(sys, dict(TRUTH, acc_ra=0.0, acc_dec=0.0))
    dr4.centroid_pos_al = plot._model_along_scan_dr4(no_acc, sys, dr4) + 0.5 * dr4.dt_yr**2 * (
        TRUTH["acc_ra"] * dr4.sin_scan + TRUTH["acc_dec"] * dr4.cos_scan
    )

    idx = [sys.param_idx["acc_ra"], sys.param_idx["acc_dec"]]
    base = plot._model_along_scan_dr4(no_acc, sys, dr4)
    design = np.column_stack(
        [plot._model_along_scan_dr4(no_acc + np.eye(len(no_acc))[i], sys, dr4) - base for i in idx]
    )
    best, *_ = np.linalg.lstsq(design, dr4.centroid_pos_al - base, rcond=None)

    np.testing.assert_allclose(best, [TRUTH["acc_ra"], TRUTH["acc_dec"]], rtol=0, atol=1e-10)


def test_relative_astrometry_ignores_acceleration(tmp_path):
    """Relative astrometry of the companion does not depend on the star's
    acceleration, so compute_model is unchanged for those rows, while the
    DR4 likelihood changes."""
    sys = kep_system(mock_dr4(), data_table=radec_table(tmp_path, obj=1))
    params = params_from(sys, TRUTH)
    no_acc = params_from(sys, dict(TRUTH, acc_ra=0.0, acc_dec=0.0))

    np.testing.assert_allclose(sys.compute_model(params)[0], sys.compute_model(no_acc)[0])
    assert lnlike(sys, params) != lnlike(sys, no_acc)


def test_hipparcos_plus_dr4():
    """Hipparcos IAD can be fitted along with DR4. The acceleration is
    extrapolated to the Hipparcos epochs, so it changes the likelihood."""
    sys = kep_system(mock_dr4(), hipparcos_IAD=hip_iad())

    assert "acc_ra" in sys.labels and "acc_dec" in sys.labels
    values = dict(
        TRUTH,
        pm_ra=sys.hipparcos_IAD.pm_ra0,
        pm_dec=sys.hipparcos_IAD.pm_dec0,
        plx=sys.hipparcos_IAD.plx0,
        alpha0=0.0,
        delta0=0.0,
    )
    params = params_from(sys, values)
    no_acc = params_from(sys, dict(values, acc_ra=0.0, acc_dec=0.0))

    assert np.isfinite(lnlike(sys, params))
    assert lnlike(sys, params) != lnlike(sys, no_acc)


def test_hipparcos_stellar_astrometry_and_dr4(tmp_path):
    """Hipparcos, stellar ra/dec astrometry and DR4 together: the System sets
    up and gives a finite likelihood."""
    sys = kep_system(
        mock_dr4(), data_table=radec_table(tmp_path, obj=0), hipparcos_IAD=hip_iad()
    )
    assert np.isfinite(lnlike(sys, params_from(sys, TRUTH)))


@pytest.mark.xfail(
    raises=AttributeError,
    strict=True,
    reason=(
        "System takes the reference RA/Dec for stellar ra/dec data from "
        "hipparcos_IAD, so stellar astrometry + DR4 without Hipparcos fails"
    ),
)
def test_stellar_astrometry_and_dr4_without_hipparcos(tmp_path):
    """Stellar ra/dec astrometry from any instrument plus DR4, with no
    Hipparcos IAD."""
    sys = kep_system(mock_dr4(), data_table=radec_table(tmp_path, obj=0))
    assert np.isfinite(lnlike(sys, params_from(sys, TRUTH)))


def test_mcmc_runs():
    """A short MCMC run works and returns finite log-likelihoods."""
    sys = kep_system(mock_dr4())
    n_walkers = 2 * len(sys.labels)
    mcmc = sampler.MCMC(sys, num_temps=1, num_walkers=n_walkers, num_threads=1)
    mcmc.curr_pos = params_from(sys, TRUTH) + 1e-4 * np.random.default_rng(0).normal(
        size=(1, n_walkers, len(sys.labels))
    )
    mcmc.run_sampler(n_walkers * 5, burn_steps=0)

    assert mcmc.results.post.shape == (n_walkers * 5, len(sys.labels))
    assert np.all(np.isfinite(mcmc.results.lnlike))


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    test_labels()
    test_needs_dr4(tmp)
    test_zero_acceleration_matches_keplerian_fit()
    test_acceleration_adds_to_star_only()
    test_acceleration_in_along_scan_model()
    test_fitted_acceleration_recovers_injected()
    test_relative_astrometry_ignores_acceleration(tmp)
    test_hipparcos_plus_dr4()
    test_hipparcos_stellar_astrometry_and_dr4(tmp)
    test_mcmc_runs()


def test_user_priors_replace_plx_and_mass():
    """plx_prior and stellar_or_system_mass_prior can be used without plx and
    stellar_or_system_mass, and take priority over them."""
    plx_prior = priors.GaussianPrior(20.0, 0.5)
    mass_prior = priors.GaussianPrior(1.0, 0.1)
    kwargs = dict(
        num_secondary_bodies=1, data_table=None, fit_secondary_mass=True, gaia=mock_dr4(),
        plx_prior=plx_prior, stellar_or_system_mass_prior=mass_prior,
    )
    for sys in (
        system.System(**kwargs),
        system.System(stellar_or_system_mass=2.0, plx=5.0, plx_err=1.0, mass_err=0.5, **kwargs),
    ):
        assert sys.sys_priors[sys.param_idx["plx"]] is plx_prior
        assert sys.sys_priors[sys.param_idx["m0"]] is mass_prior


def test_total_mass_prior(tmp_path):
    """Without a fitted companion mass, the prior is on the total mass."""
    table = radec_table(tmp_path, obj=1)
    mass_prior = priors.GaussianPrior(1.0, 0.1)
    sys = system.System(1, table, plx=20.0, stellar_or_system_mass_prior=mass_prior)
    assert sys.sys_priors[sys.param_idx["mtot"]] is mass_prior


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(plx=20.0), "stellar_or_system_mass"),
        (dict(stellar_or_system_mass=1.0), "plx"),
    ],
)
def test_needs_plx_and_mass(tmp_path, kwargs, match):
    """A Keplerian fit needs a value or a prior for both plx and mass."""
    with pytest.raises(ValueError, match=match):
        system.System(1, radec_table(tmp_path, obj=1), **kwargs)


def test_save_system_without_plx_or_mass(tmp_path):
    """A System set up with only priors can still be saved."""
    sys = system.System(
        num_secondary_bodies=1, data_table=None, fit_secondary_mass=True, gaia=mock_dr4(),
        plx_prior=priors.GaussianPrior(20.0, 0.5),
        stellar_or_system_mass_prior=priors.GaussianPrior(1.0, 0.1),
    )
    with h5py.File(tmp_path / "system.hdf5", "w") as hf:
        sys.save(hf)
        assert np.isnan(hf.attrs["plx"])
        assert np.isnan(hf.attrs["stellar_or_system_mass"])


@pytest.mark.parametrize("fit", ["keplerian+acc", "keplerian", "acc_only"])
def test_plot_dr4_acceleration(fit):
    """plot_dr4_acceleration works for all three kinds of fit by default and
    labels every y axis."""
    dr4 = mock_dr4()
    if fit == "acc_only":
        sys = system.System(num_secondary_bodies=0, data_table=None, gaia=dr4, acceleration_only=True)
    else:
        sys = kep_system(dr4) if fit == "keplerian+acc" else system.System(
            num_secondary_bodies=1, data_table=None, stellar_or_system_mass=1.0, mass_err=0.1,
            plx=20.0, plx_err=0.5, fit_secondary_mass=True, gaia=dr4,
        )
    assert ("acc_ra" in sys.labels) == (fit != "keplerian")

    values = dict(TRUTH, acc_ra=0.05, acc_dec=-0.04, plx=20.0)
    post = np.tile(params_from(sys, values), (10, 1))
    post += 1e-3 * np.random.default_rng(0).normal(size=post.shape)
    results = SimpleNamespace(post=post)

    plt.switch_backend("Agg")
    fig = plot.plot_dr4_acceleration(results, sys, num_orbits_to_plot=5, num_epochs_to_plot=20)
    axes = fig.get_axes()[:6]
    assert all(ax.get_ylabel().endswith("[mas]") for ax in axes)
    assert axes[-1].get_ylabel() == ("orbit [mas]" if fit == "keplerian+acc" else "residuals [mas]")
    plt.close(fig)
