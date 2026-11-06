"""
Every supported combination of data and fit type should set up a System,
give a finite likelihood and run the MCMC sampler; every unsupported one
should raise a clear ValueError instead of failing later.

Data:
    dr4      Gaia DR4 epoch astrometry (DR4LogProb, through ``gaia``)
    abs      stellar ra/dec astrometry from any instrument (object 0)
    rel      relative astrometry of the companion (object 1)
    hip      Hipparcos IAD

Fit types:
    keplerian         Keplerian orbit
    keplerian+acc     acceleration_and_keplerian_fit=True
    acc_only          acceleration_only=True
"""

import os

import numpy as np
import pytest
from astropy.table import Table
from astropy.time import Time

from orbitize import DATADIR, gaia, hipparcos, read_input, results, sampler, system

REF_EPOCH_MJD = 57936.375  # J2017.5, the DR4 reference epoch [MJD]
ALPHA0_DEG, DELTA0_DEG = 86.82, -51.07  # reference RA/Dec for acc_only ra/dec data

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

BUG_ABS_DR4 = pytest.mark.xfail(
    raises=AttributeError,
    strict=True,
    reason=(
        "System takes the reference RA/Dec for stellar ra/dec data from "
        "hipparcos_IAD, so stellar astrometry + DR4 without Hipparcos fails"
    ),
)

# (fit type, data) -> "ok" or the exception the System should raise
CASES = [
    ("acc_only", "dr4", "ok"),
    ("acc_only", "abs", ValueError),  # ra/dec come from the Gaia object
    ("acc_only", "abs+dr4", "ok"),
    ("acc_only", "rel+dr4", ValueError),
    ("acc_only", "hip+dr4", ValueError),
    ("acc_only", "hip+abs+dr4", ValueError),
    ("keplerian+acc", "dr4", "ok"),
    ("keplerian+acc", "abs", ValueError),
    ("keplerian+acc", "rel", ValueError),
    pytest.param("keplerian+acc", "abs+dr4", "ok", marks=BUG_ABS_DR4),
    ("keplerian+acc", "rel+dr4", "ok"),
    ("keplerian+acc", "hip+dr4", "ok"),
    ("keplerian+acc", "hip+abs+dr4", "ok"),
    ("keplerian+acc", "hip+rel+dr4", "ok"),
    ("keplerian", "dr4", "ok"),
    pytest.param("keplerian", "abs+dr4", "ok", marks=BUG_ABS_DR4),
    ("keplerian", "rel+dr4", "ok"),
    ("keplerian", "hip+dr4", "ok"),
    ("keplerian", "hip+abs+dr4", "ok"),
]


def mock_dr4(n_obs=30, err=0.05, seed=0):
    """Mock Gaia DR4 along-scan data over 5 yr centred on J2017.5."""
    rng = np.random.default_rng(seed)
    epochs = REF_EPOCH_MJD + np.linspace(-2.5, 2.5, n_obs) * 365.25
    data = Table(
        {
            "obs_time_tcb": epochs,
            "centroid_pos_al": rng.normal(0, err, n_obs),
            "centroid_pos_error_al": np.full(n_obs, err),
            "scan_pos_angle": rng.uniform(0, 2 * np.pi, n_obs),
            "parallax_factor_al": np.sin(2 * np.pi * (epochs - REF_EPOCH_MJD) / 365.25),
            "field_of_view": np.zeros(n_obs, dtype=int),
        }
    )
    data.meta.update(time_format="mjd", time_scale="tcb")
    return gaia.DR4LogProb(
        0, data, ref_epoch_mjd=REF_EPOCH_MJD, ra=ALPHA0_DEG, dec=DELTA0_DEG
    )


def astrometry_table(tmp_path, objects):
    """Read ra/dec astrometry rows for each object in ``objects`` (0 = star,
    1 = companion) from a csv file."""
    years = np.array([2015.0, 2017.0, 2018.5, 2020.0])
    rows = []
    for obj in objects:
        value = 50.0 if obj else 0.0  # companion offset [mas]; star at 0
        for epoch in Time(years, format="jyear").mjd:
            rows.append((epoch, obj, value, 0.5, value, 0.5))
    t = Table(
        rows=rows,
        names=["epoch", "object", "raoff", "raoff_err", "decoff", "decoff_err"],
    )
    filename = tmp_path / "astrometry.csv"
    t.write(filename, overwrite=True)
    return read_input.read_file(str(filename))


def make_system(fit_type, data, tmp_path):
    parts = data.split("+")
    objects = [obj for name, obj in (("abs", 0), ("rel", 1)) if name in parts]
    kwargs = dict(
        data_table=astrometry_table(tmp_path, objects) if objects else None,
        gaia=mock_dr4() if "dr4" in parts else None,
        hipparcos_IAD=(
            hipparcos.HipparcosLogProb(
                "{}HIP027321.d".format(DATADIR), "027321", 1, hip_data=HIP_DATA
            )
            if "hip" in parts
            else None
        ),
    )
    if fit_type == "acc_only":
        return system.System(
            num_secondary_bodies=0,
            stellar_or_system_mass=1.0,
            plx=10.0,
            acceleration_only=True,
            **kwargs,
        )
    return system.System(
        num_secondary_bodies=1,
        stellar_or_system_mass=1.0,
        plx=20.0,
        mass_err=0.1,
        plx_err=0.5,
        fit_secondary_mass=True,
        acceleration_and_keplerian_fit=(fit_type == "keplerian+acc"),
        **kwargs,
    )


def draw_params(sys, n, seed=0):
    """n parameter vectors drawn from the System's priors (n x n_params)."""
    np.random.seed(seed)
    return np.array(
        [
            prior.draw_samples(n)
            if hasattr(prior, "draw_samples")
            else np.full(n, prior)
            for prior in sys.sys_priors
        ]
    ).T


@pytest.mark.parametrize("fit_type, data, expected", CASES)
def test_data_combination(tmp_path, fit_type, data, expected):
    """Set up the System, check the likelihood is finite for draws from the
    priors, and run a few MCMC steps."""
    if expected != "ok":
        with pytest.raises(expected):
            make_system(fit_type, data, tmp_path)
        return

    sys = make_system(fit_type, data, tmp_path)
    assert len(sys.sys_priors) == len(sys.labels)
    assert ("acc_ra" in sys.labels) == (fit_type != "keplerian")

    n_walkers = 2 * len(sys.labels)
    mcmc = sampler.MCMC(sys, num_temps=1, num_walkers=n_walkers, num_threads=1)
    start = draw_params(sys, n_walkers)
    assert all(np.isfinite(mcmc._logl(p)) for p in start)

    mcmc.curr_pos = start[None]
    mcmc.run_sampler(n_walkers * 3, burn_steps=0)
    assert np.all(np.isfinite(mcmc.results.lnlike))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Results.load_results does not rebuild a DR4LogProb or the "
        "acceleration_only / acceleration_and_keplerian_fit settings"
    ),
)
@pytest.mark.parametrize("fit_type", ["acc_only", "keplerian+acc", "keplerian"])
def test_save_load_dr4_results(tmp_path, fit_type):
    """Saving and reloading results from a DR4 fit gives back the same
    System: the same labels and a DR4LogProb likelihood."""
    sys = make_system(fit_type, "dr4", tmp_path)
    n_walkers = 2 * len(sys.labels)
    mcmc = sampler.MCMC(sys, num_temps=1, num_walkers=n_walkers, num_threads=1)
    mcmc.curr_pos = draw_params(sys, n_walkers)[None]
    mcmc.run_sampler(n_walkers * 2, burn_steps=0)

    filename = os.path.join(tmp_path, "dr4_results.hdf5")
    mcmc.results.save_results(filename)
    loaded = results.Results()
    loaded.load_results(filename)

    assert loaded.labels == sys.labels
    assert isinstance(loaded.system.gaia, gaia.DR4LogProb)
    assert loaded.system.acceleration_only == sys.acceleration_only
    assert (
        loaded.system.acceleration_and_keplerian_fit
        == sys.acceleration_and_keplerian_fit
    )
