"""
Simulate Gaia DR4 epoch astrometry (one along-scan measurement per transit).

    from orbitize.simulate_dr4 import simulate_from_solution
    simulate_from_solution("injected_solutions.csv", target="target_1.csv")

or from the command line::

    python -m orbitize.simulate_dr4 injected_solutions.csv --target target_1.csv

The output loads straight into ``orbitize.gaia.DR4LogProb``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table

from orbitize import DATADIR, kepler

REF_EPOCH_MJD = 57936.375  # J2017.5, DR4LogProb default
TAU_REF_EPOCH = 58849  # orbitize.system.System default
DR4_MJD_RANGE = (56863.4, 58868.9)  # 2014-07-25 to 2020-01-20
MJUP = (1 * u.Mjup).to_value(u.Msun)

# Must match the sign of the orbit term in DR4LogProb.compute_lnlike.
ORBIT_AL_SIGN = -1.0

GOST_COLUMNS = {
    "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": "obs_time_tcb",
    "scanAngle[rad]": "scan_pos_angle",
    "parallaxFactorAlongScan": "parallax_factor_al",
    "Fov[FovP=preceding/FovF=following]": "field_of_view",
}
G_GRID = [4, 5, 6, 7, 8.2, 8.4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
SIGMA_CCD_MAS = [0.40, 0.35, 0.15, 0.17, 0.23, 0.13, 0.13, 0.135, 0.125,
                 0.13, 0.15, 0.23, 0.36, 0.63, 1.05, 2.05, 4.10]


def read_gost(gost):
    """Scanning law (GOST CSV, DR4-format file, dict, DataFrame or Table) -> Table in MJD."""
    t = Table.read(gost, format="ascii.csv") if isinstance(gost, (str, Path)) else Table(dict(gost))
    t.rename_columns(t.colnames, [GOST_COLUMNS.get(c.strip(), c.strip()) for c in t.colnames])
    t["obs_time_tcb"] = np.asarray(t["obs_time_tcb"], float)
    if t["obs_time_tcb"][0] > 2400000.5:  # JD -> MJD
        t["obs_time_tcb"] -= 2400000.5
    if "field_of_view" in t.colnames and t["field_of_view"].dtype.kind in "US":
        t["field_of_view"] = np.char.strip(t["field_of_view"].astype(str)) == "FovF"
        t["field_of_view"] = t["field_of_view"].astype(int) + 1  # FovP -> 1, FovF -> 2
    keep = (t["obs_time_tcb"] >= DR4_MJD_RANGE[0]) & (t["obs_time_tcb"] <= DR4_MJD_RANGE[1])
    return t[keep]


def star_orbit(epochs, p):
    """Star's offset from the barycentre [mas], summed over companions 1, 2, ..."""
    ra, dec = np.zeros(len(epochs)), np.zeros(len(epochs))
    n = 0
    while "sma{}".format(n + 1) in p:
        n += 1
    smas = np.array([p["sma{}".format(i)] for i in range(1, n + 1)])
    masses = np.array([p["m{}".format(i)] for i in range(1, n + 1)])
    for i in range(1, n + 1):
        # Same enclosed-mass rule as System.compute_all_orbits
        mtot = p["m0"] + masses[smas <= p["sma{}".format(i)]].sum()
        r, d = kepler.calc_orbit(
            epochs, *(p["{}{}".format(k, i)] for k in ("sma", "ecc", "inc", "aop", "pan", "tau")),
            p["plx"], mtot, tau_ref_epoch=p.get("tau_ref_epoch", TAU_REF_EPOCH))[:2]
        ra -= p["m{}".format(i)] / mtot * r
        dec -= p["m{}".format(i)] / mtot * d
    return ra, dec


def simulate_dr4(gost, params, g_mag=None, sigma_al=None, seed=None,
                 add_noise=True, jitter=0.0, output_file=None):
    """Simulate DR4 along-scan data for one star.

    Args:
        gost: scanning law; see read_gost. If it has centroid_pos_error_al and
            neither g_mag nor sigma_al is given, those uncertainties are reused.
        params (dict): orbitize standard-basis values: plx [mas]; alpha0, delta0
            [mas]; pm_ra, pm_dec [mas/yr]; and per companion i: sma{i} [au],
            ecc{i}, inc{i}, aop{i}, pan{i} [rad], tau{i}, m{i} plus m0 [Msun].
        g_mag / sigma_al: G magnitude, or per-transit uncertainty [mas].
        jitter: extra noise [mas] not included in the reported uncertainties.
        output_file: optional .csv (JD times) or .ecsv (keeps units and truth).

    Returns:
        astropy.table.Table in the format DR4LogProb reads.
    """
    scans = read_gost(gost)
    t = np.asarray(scans["obs_time_tcb"])
    sin, cos = np.sin(scans["scan_pos_angle"]), np.cos(scans["scan_pos_angle"])
    dt = (t - REF_EPOCH_MJD) / 365.25

    if sigma_al is None and g_mag is None:
        sigma = np.asarray(scans["centroid_pos_error_al"], float)
    elif sigma_al is None:  # per-CCD curve, 8 CCDs per transit
        sigma = np.full(len(t), np.interp(g_mag, G_GRID, SIGMA_CCD_MAS) / np.sqrt(8))
    else:
        sigma = np.broadcast_to(sigma_al, t.shape).astype(float)

    p = {"alpha0": 0.0, "delta0": 0.0, "pm_ra": 0.0, "pm_dec": 0.0, **params}
    ra, dec = star_orbit(t, p)
    al = ((p["alpha0"] + p["pm_ra"] * dt + ORBIT_AL_SIGN * ra) * sin
          + (p["delta0"] + p["pm_dec"] * dt + ORBIT_AL_SIGN * dec) * cos
          + p["plx"] * np.asarray(scans["parallax_factor_al"]))
    if add_noise:
        al = al + np.random.default_rng(seed).normal(0, np.hypot(sigma, jitter))

    out = Table({
        "obs_time_tcb": t * u.day,
        "centroid_pos_al": al * u.mas,
        "centroid_pos_error_al": sigma * u.mas,
        "parallax_factor_al": np.asarray(scans["parallax_factor_al"], float),
        "scan_pos_angle": np.asarray(scans["scan_pos_angle"], float) * u.rad,
        "field_of_view": scans["field_of_view"] if "field_of_view" in scans.colnames else np.zeros(len(t), int),
    }, meta={"time_format": "mjd", "time_scale": "tcb", "truth": p, "seed": seed})

    if output_file is not None:
        if str(output_file).endswith(".ecsv"):
            out.write(output_file, overwrite=True)
        else:  # plain CSV: JD times, the format orbitize assumes for CSV input
            flat = out.copy()
            flat["obs_time_tcb"] = t + 2400000.5
            flat.meta.clear()
            flat.write(output_file, format="ascii.csv", overwrite=True)
    return out


def params_from_solution(row):
    """Solutions-table row (injected_solutions.csv columns) -> simulate_dr4 params."""
    p = {"plx": row["parallax_mas"], "alpha0": row["alpha_mas"], "delta0": row["delta_mas"],
         "pm_ra": row["mu_alpha_mas_yr"], "pm_dec": row["mu_delta_mas_yr"]}
    if pd.isna(row.get("P_d")):  # single star
        return p
    P_yr, m_b = row["P_d"] / 365.25, row["m_b"] * MJUP
    # a0 = plx * a * m_b / M and a^3 = P^2 M  ->  solve for total mass M
    m_tot = (row["parallax_mas"] * m_b * P_yr ** (2 / 3) / row["a0_mas"]) ** 1.5
    p.update(
        sma1=(P_yr ** 2 * m_tot) ** (1 / 3), ecc1=row["e"], inc1=np.radians(row["i_deg"]),
        # solution angles describe the star's orbit; orbitize's describe the companion's,
        # and the -1 likelihood sign adds another 180 deg to pan
        aop1=np.radians(row["omega_deg"] + 180) % (2 * np.pi),
        pan1=np.radians(row["big_omega_deg"] + (180 if ORBIT_AL_SIGN < 0 else 0)) % (2 * np.pi),
        tau1=(-row["M0_deg"] / 360 + (REF_EPOCH_MJD - TAU_REF_EPOCH) / row["P_d"]) % 1,
        m1=m_b, m0=m_tot - m_b,
    )
    return {k: float(v) for k, v in p.items()}


def params_from_sample(system, sample, **overrides):
    """One posterior sample (e.g. best RV fit) -> simulate_dr4 params.

    Overrides of fitting-basis parameters (e.g. inc1=np.pi/2) are applied before
    converting to the standard basis, so derived masses stay consistent.
    """
    sample = np.array(sample, float)
    for name in [k for k in overrides if k in system.param_idx]:
        sample[system.param_idx[name]] = overrides.pop(name)
    std, idx = system.basis.to_standard_basis(sample.copy()), system.basis.standard_basis_idx
    orbit_keys = ("sma", "ecc", "inc", "aop", "pan", "tau", "m")  # drops RV offsets/jitters
    p = {k: std[i] for k, i in idx.items() if k == "plx" or k.rstrip("0123456789") in orbit_keys}
    p.update({k: sample[system.param_idx[k]] for k in ("alpha0", "delta0", "pm_ra", "pm_dec")
              if k in system.param_idx})
    p.update(tau_ref_epoch=system.tau_ref_epoch, **overrides)
    return {k: float(v) for k, v in p.items()}


def simulate_from_solution(solutions_file, target=None, scanning_law=None,
                           output_file=None, **kwargs):
    """One row of a solutions CSV -> simulated DR4 CSV.

    The row's ``filename`` names the scanning law, looked for next to the
    solutions file and in orbitize's example_data. Output defaults to
    ``<filename>_simulated_dr4.csv``. kwargs go to simulate_dr4 (seed, g_mag, ...).
    """
    rows = pd.read_csv(solutions_file)
    if target is not None:
        rows = rows[rows["filename"].str.strip() == target]
    if len(rows) != 1:
        raise ValueError("Pick one row with target=, e.g. target='target_1.csv'.")
    row = rows.iloc[0].to_dict()
    name = row["filename"].strip()
    if scanning_law is None:
        folders = [Path(solutions_file).parent, Path(DATADIR)]
        scanning_law = next(f / name for f in folders if (f / name).exists())
    output_file = output_file or "{}_simulated_dr4.csv".format(Path(name).stem)
    return simulate_dr4(scanning_law, params_from_solution(row), output_file=output_file, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Gaia DR4 data from a solutions-table row.")
    parser.add_argument("solutions")
    parser.add_argument("--target")
    parser.add_argument("--scan")
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--g-mag", type=float)
    a = parser.parse_args()
    table = simulate_from_solution(a.solutions, a.target, a.scan, a.out, seed=a.seed, g_mag=a.g_mag)
    print("Wrote {} transits".format(len(table)))
