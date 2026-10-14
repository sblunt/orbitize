import os

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from orbitize import DATADIR, gaia, read_input, system
from orbitize import simulate_dr4 as sim

PARAMS = dict(sma1=3.4, ecc1=0.07, inc1=1.0, aop1=1.1, pan1=2.2, tau1=0.4, m1=1e-3,
              m0=0.82, plx=310.6, alpha0=0.3, delta0=-0.2, pm_ra=-974.8, pm_dec=20.9)


def fake_gost(n=80):
    rng = np.random.default_rng(0)
    return {"obs_time_tcb": np.sort(rng.uniform(56900, 58800, n)) + 2400000.5,
            "scan_pos_angle": rng.uniform(-np.pi, np.pi, n),
            "parallax_factor_al": rng.uniform(-0.7, 0.7, n),
            "field_of_view": rng.choice(["FovP", "FovF"], n)}


def test_noiseless_data_fit_the_likelihood_exactly():
    params = dict(PARAMS, sma2=9.0, ecc2=0.3, inc2=0.4, aop2=0.2, pan2=5.0, tau2=0.8, m2=3e-3)
    table = sim.simulate_dr4(fake_gost(), params, sigma_al=0.05, add_noise=False)
    empty = Table(names=("epoch", "object", "quant1", "quant1_err", "quant2", "quant2_err",
                         "quant12_corr", "quant_type", "instrument"),
                  dtype=(float, int, float, float, float, float, float, "U5", "U8"))
    s = system.System(2, empty, 0.82, 310.6, mass_err=0.02, plx_err=0.1,
                      fit_secondary_mass=True, gaia=gaia.DR4LogProb(0, table))
    truth = np.array([params[name] for name in s.param_idx])
    ra, dec = s.compute_all_orbits(truth, epochs=s.gaia.epochs_mjd)[:2]
    lnlike = s.gaia.compute_lnlike(ra[:, 0, :], dec[:, 0, :], truth, s.param_idx)
    assert lnlike == pytest.approx(-0.5 * len(table) * np.log(2 * np.pi * 0.05**2))


def test_noise_level_and_fov():
    table = sim.simulate_dr4(fake_gost(4000), PARAMS, sigma_al=0.05, seed=3)
    model = sim.simulate_dr4(fake_gost(4000), PARAMS, sigma_al=0.05, add_noise=False)
    assert np.std(np.asarray(table["centroid_pos_al"]) - np.asarray(model["centroid_pos_al"])) == pytest.approx(0.05, rel=0.05)
    assert set(table["field_of_view"]) == {1, 2}


@pytest.mark.parametrize("name", ["mock.csv", "mock.ecsv"])
def test_output_file_reads_back(tmp_path, name):
    table = sim.simulate_dr4(fake_gost(), PARAMS, g_mag=10, seed=1, output_file=tmp_path / name)
    dr4 = gaia.DR4LogProb(0, str(tmp_path / name))
    np.testing.assert_allclose(dr4.epochs_mjd, table["obs_time_tcb"], atol=1e-6)
    np.testing.assert_allclose(dr4.centroid_pos_al, table["centroid_pos_al"])


def test_solution_row_reproduces_mock_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solutions = os.path.join(DATADIR, "injected_solutions.csv")
    table = sim.simulate_from_solution(solutions, "target_1.csv", add_noise=False)
    assert (tmp_path / "target_1_simulated_dr4.csv").exists()
    orig = pd.read_csv(os.path.join(DATADIR, "target_1.csv"))
    resid = (orig.centroid_pos_al - np.asarray(table["centroid_pos_al"])) / orig.centroid_pos_error_al
    assert np.mean(resid**2) < 1.5  # matches to within the mock's own noise


def test_single_star_solution():
    row = dict(parallax_mas=31.4, alpha_mas=0.0, delta_mas=0.0,
               mu_alpha_mas_yr=-24.8, mu_delta_mas_yr=-38.8, P_d=np.nan)
    assert "sma1" not in sim.params_from_solution(row)


def test_params_from_sample_rv_fit():
    """SemiAmp (RV) sample with injected orientation, as in the eps Eri notebook."""
    rv = read_input.read_file(os.path.join(DATADIR, "HD4747.csv"))
    s = system.System(1, rv, 0.82, 310.6, mass_err=0.02, plx_err=0.1,
                      fit_secondary_mass=True, fitting_basis="SemiAmp")
    sample = np.array([prior.draw_samples(1)[0] if hasattr(prior, "draw_samples") else prior
                       for prior in s.sys_priors])
    p = sim.params_from_sample(s, sample, inc1=np.radians(40), pan1=np.radians(186),
                               pm_ra=-974.8, pm_dec=20.9)
    assert p["inc1"] == pytest.approx(np.radians(40)) and p["pm_ra"] == -974.8
    assert not any(k.startswith(("gamma", "sigma")) for k in p)
    sample[s.param_idx["inc1"]], sample[s.param_idx["pan1"]] = np.radians(40), np.radians(186)
    std = s.basis.to_standard_basis(sample.copy())
    assert p["m1"] == pytest.approx(std[s.basis.standard_basis_idx["m1"]])
    sim.simulate_dr4(fake_gost(), p, g_mag=3.47, seed=1)
