import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.table import Table

from orbitize import simulate_dr4 as sim


def _write_gost(tmp_path, epochs, scan_angles, parallax_factors):
    """Create a tiny GOST file with JD(TCB) timestamps."""
    path = tmp_path / "gost.csv"
    Table(
        {
            "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": (
                np.asarray(epochs) + 2400000.5
            ),
            "scanAngle[rad]": scan_angles,
            "parallaxFactorAlongScan": parallax_factors,
            "Fov[FovP=preceding/FovF=following]": ["FoVP"] * len(epochs),
        }
    ).write(path, format="ascii.csv")
    return path


@pytest.fixture
def gost_file(tmp_path):
    ref = sim.DR4_REF_EPOCH_MJD
    return _write_gost(
        tmp_path,
        [ref - 365.25, ref, ref + 365.25],
        [0.0, np.pi / 2, 0.0],
        [0.1, -0.2, 0.3],
    )


def test_simulate_dr4_linear_model(gost_file, monkeypatch):
    """Position, proper motion, and parallax give known AL positions."""
    # make random noise 0
    monkeypatch.setattr(sim, "al_uncertainty", lambda g_mag: 0.0)

    # make ra0, dec0, pmra, pmdec, plx
    params = dict(
        alpha0=2.0,
        delta0=-3.0,
        pm_ra=0.5,
        pm_dec=-0.25,
        plx=20.0,
    )
    table = sim.simulate_dr4(gost_file, params, seed=123)

    # dt = [-1, 0, 1] yr. scan angles = [0, pi/2, 0].
    np.testing.assert_allclose(
        table["centroid_pos_al"],
        [-0.75, -2.0, 2.75],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(table["centroid_pos_error_al"], np.zeros(3))

    assert table["obs_time_tcb"].unit == u.day
    assert table["centroid_pos_al"].unit == u.mas
    assert table["centroid_pos_error_al"].unit == u.mas
    assert table.meta["time_format"] == "mjd"
    assert table.meta["time_scale"] == "tcb"


def test_simulate_dr4_stellar_reflex(tmp_path, monkeypatch):
    """A known circular orbit gives the correct stellar reflex signal."""
    monkeypatch.setattr(sim, "al_uncertainty", lambda g_mag: 0.0)

    # period to sma
    period_days = np.sqrt(4 * np.pi**2 * u.au**3 / (const.G * u.Msun)).to_value(u.day)

    # make mock gost file
    ref = sim.DR4_REF_EPOCH_MJD
    path = _write_gost(
        tmp_path,
        ref + period_days * np.array([0.0, 0.25, 0.5]),
        [0.0, np.pi / 2, 0.0],
        [0.0, 0.0, 0.0],
    )

    # make arbitrary orbit (face on, circular)
    params = dict(
        sma1=1.0,
        ecc1=0.0,
        inc1=0.0,
        aop1=0.0,
        pan1=0.0,
        tau1=0.0,
        plx=20.0,
        m0=0.99,
        m1=0.01,
    )
    # make mock dr4 data
    table = sim.simulate_dr4(path, params, seed=123, tau_ref_epoch=ref)

    # Relative orbit radius: 20 mas.
    # Stellar reflex radius: 0.01 * 20 = 0.2 mas.
    # Expected values include the along scan sign convention.
    np.testing.assert_allclose(
        table["centroid_pos_al"],
        [-0.2, -0.2, 0.2],
        rtol=0,
        atol=1e-9,
    )
