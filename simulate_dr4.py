"""
Simulate Gaia DR4 epoch astrometry (one along-scan measurement per transit).

Intended home: ``orbitize/gaia.py`` (or a new ``orbitize/simulate.py``) on the
``gaia-dr4-module`` branch. The output table is in the same transit-level format
that ``orbitize.gaia.DR4LogProb`` consumes, so it can be passed straight in::

    table = simulate_dr4("my_star_gost.csv", params, g_mag=6.2, seed=1)
    dr4 = gaia.DR4LogProb(0, table, catalog_pmra=..., catalog_pmdec=...)

Forward model per transit (identical to ``DR4LogProb.compute_lnlike``)::

    AL_i = alpha0 * sin(psi_i) + delta0 * cos(psi_i)
         + pm_ra * dt_i * sin(psi_i) + pm_dec * dt_i * cos(psi_i)
         + plx * parallax_factor_i
         + ORBIT_AL_SIGN * (ra_star_i * sin(psi_i) + dec_star_i * cos(psi_i))
         + noise_i
"""

import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.time import Time

from orbitize import kepler

__all__ = ["simulate_dr4", "write_dr4", "read_gost", "params_from_sample", "al_uncertainty"]

DR4_REF_EPOCH_MJD = 57936.375  # J2017.5, MJD(TCB); DR4LogProb default
DEFAULT_TAU_REF_EPOCH = 58849  # orbitize.system.System default

# Time span of the data that go into DR4 (66 months).
DR4_START_MJD = Time("2014-07-25T10:30:00", scale="tcb").mjd
DR4_END_MJD = Time("2020-01-20T22:00:00", scale="tcb").mjd

# Sign applied to the star's barycentric orbit when projecting it onto the scan
# direction. -1 reproduces DR4LogProb.compute_lnlike as it is currently written
# on gaia-dr4-module. If that sign is changed in the likelihood, change it here
# too (test_matches_likelihood fails if the two disagree).
ORBIT_AL_SIGN = -1.0

# Per-CCD along-scan uncertainty vs G magnitude [mas] (from the eps Eri mock).
_G_GRID = np.array(
    [4, 5, 6, 7, 8.2, 8.4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
)
_SIGMA_CCD_GRID_MAS = np.array(
    [0.40, 0.35, 0.15, 0.17, 0.23, 0.13, 0.13, 0.135, 0.125,
     0.13, 0.15, 0.23, 0.36, 0.63, 1.05, 2.05, 4.10]
)

# Raw GOST header -> orbitize DR4 column name
_GOST_COLUMNS = {
    "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": "obs_time_tcb",
    "scanAngle[rad]": "scan_pos_angle",
    "parallaxFactorAlongScan": "parallax_factor_al",
    "Fov[FovP=preceding/FovF=following]": "field_of_view",
}
_SCAN_COLUMNS = ("obs_time_tcb", "scan_pos_angle", "parallax_factor_al")
_LINEAR_PARAMS = ("alpha0", "delta0", "pm_ra", "pm_dec")
_ORBIT_PARAMS = ("sma", "ecc", "inc", "aop", "pan", "tau", "m")


def read_gost(gost, clip_to_dr4=True):
    """Read a Gaia scanning law into a table of transits.

    Args:
        gost (str, pathlib.Path, dict, pandas.DataFrame, or astropy.table.Table):
            Either a CSV downloaded from https://gaia.esac.esa.int/gost/ (raw
            GOST headers are recognised), or any table-like object / dict of
            equal-length arrays with the keys ``obs_time_tcb`` (JD or MJD, TCB),
            ``scan_pos_angle`` [rad], ``parallax_factor_al`` and, optionally,
            ``field_of_view``.
        clip_to_dr4 (bool): drop transits outside the DR4 time span
            (2014-07-25 to 2020-01-20). GOST returns whatever range was
            queried, which is often the whole mission.

    Returns:
        astropy.table.Table: ``obs_time_tcb`` [MJD, TCB], ``scan_pos_angle``
        [rad], ``parallax_factor_al``, ``field_of_view``; sorted by time.
    """
    if isinstance(gost, (str, Path)):
        table = Table.read(gost, format="ascii.csv")
    elif isinstance(gost, Table):
        table = gost.copy()
    elif hasattr(gost, "to_dict") and hasattr(gost, "columns"):  # DataFrame
        table = Table({str(c): np.asarray(gost[c]) for c in gost.columns})
    else:
        table = Table({k: np.atleast_1d(v) for k, v in dict(gost).items()})

    # GOST headers carry leading spaces
    for name in list(table.colnames):
        clean = _GOST_COLUMNS.get(name.strip(), name.strip())
        if clean != name:
            table.rename_column(name, clean)

    missing = [name for name in _SCAN_COLUMNS if name not in table.colnames]
    if missing:
        raise ValueError(
            "Scanning law is missing: {}. Found columns: {}".format(
                ", ".join(missing), ", ".join(table.colnames)
            )
        )

    out = Table()
    for name in _SCAN_COLUMNS:
        out[name] = np.asarray(table[name], dtype=float)
    if not np.all(np.isfinite(out.as_array().view(float))):
        raise ValueError("Scanning law contains non-finite values.")

    # JD(TCB) -> MJD(TCB). Gaia-era JDs are ~2.457e6 and MJDs ~5.7e4.
    if np.all(out["obs_time_tcb"] > 2400000.5):
        out["obs_time_tcb"] -= 2400000.5
    elif np.any(out["obs_time_tcb"] > 2400000.5):
        raise ValueError("obs_time_tcb mixes JD and MJD values.")

    if "field_of_view" in table.colnames:
        fov = np.asarray(table["field_of_view"])
        if fov.dtype.kind in "US":  # GOST strings -> DR4 integers
            fov = np.char.strip(fov.astype(str))
            fov = np.where(fov == "FovP", 1, np.where(fov == "FovF", 2, 0))
        out["field_of_view"] = fov.astype(int)
    else:
        out["field_of_view"] = np.zeros(len(out), dtype=int)

    if clip_to_dr4:
        keep = (out["obs_time_tcb"] >= DR4_START_MJD) & (
            out["obs_time_tcb"] <= DR4_END_MJD
        )
        if not np.all(keep):
            warnings.warn(
                "Dropped {} of {} transits outside the DR4 time span "
                "(set clip_to_dr4=False to keep them).".format(
                    np.sum(~keep), len(out)
                )
            )
        out = out[keep]
    if len(out) == 0:
        raise ValueError("No transits left in the scanning law.")

    out.sort("obs_time_tcb")
    return out


def al_uncertainty(g_mag, n_ccd=8, noise_floor=0.0):
    """Per-transit along-scan uncertainty [mas] for a star of magnitude G.

    Interpolates a per-CCD uncertainty curve, divides by sqrt(``n_ccd``), and
    adds ``noise_floor`` [mas] in quadrature. The curve covers 4 <= G <= 20 and
    is held constant outside that range.
    """
    if not _G_GRID[0] <= g_mag <= _G_GRID[-1]:
        warnings.warn(
            "g_mag={} is outside the tabulated range {:g}-{:g}; using the "
            "nearest tabulated uncertainty. Consider passing sigma_al.".format(
                g_mag, _G_GRID[0], _G_GRID[-1]
            )
        )
    sigma_ccd = np.interp(g_mag, _G_GRID, _SIGMA_CCD_GRID_MAS)
    return float(np.hypot(sigma_ccd / np.sqrt(n_ccd), noise_floor))


def _parse_params(params):
    """Validate the truth dict; return (clean dict, number of companions)."""
    clean = {k: float(v) for k, v in dict(params).items()}
    clean.setdefault("tau_ref_epoch", DEFAULT_TAU_REF_EPOCH)
    for name in _LINEAR_PARAMS:
        clean.setdefault(name, 0.0)

    n_bodies = 0
    while "sma{}".format(n_bodies + 1) in clean:
        n_bodies += 1

    required = ["plx"] + (["m0"] if n_bodies else [])
    allowed = set(_LINEAR_PARAMS) | {"plx", "m0", "tau_ref_epoch"}
    for body in range(1, n_bodies + 1):
        names = ["{}{}".format(p, body) for p in _ORBIT_PARAMS]
        required += names
        allowed |= set(names)

    missing = [name for name in required if name not in clean]
    if missing:
        raise ValueError("params is missing: {}".format(", ".join(missing)))
    unknown = sorted(set(clean) - allowed)
    if unknown:
        raise ValueError(
            "Unrecognised params: {}. Companions are numbered from 1 with no "
            "gaps; allowed keys are plx, m0, tau_ref_epoch, {}, and {} for "
            "each companion i.".format(
                ", ".join(unknown),
                ", ".join(_LINEAR_PARAMS),
                ", ".join(p + "i" for p in _ORBIT_PARAMS),
            )
        )
    if not all(np.isfinite(v) for v in clean.values()):
        raise ValueError("params contains non-finite values.")
    return clean, n_bodies


def _star_orbit(epochs_mjd, params, n_bodies):
    """Star's offset from the system barycentre [mas] in RA* and Dec.

    Mirrors ``System.compute_all_orbits(...)[:, 0]`` with
    ``fit_secondary_mass=True``: each companion's relative orbit uses the mass
    enclosed by its orbit, and the star moves by -m_i/mtot_i times that orbit.
    """
    ra_star = np.zeros(len(epochs_mjd))
    dec_star = np.zeros(len(epochs_mjd))
    smas = np.array([params["sma{}".format(b)] for b in range(1, n_bodies + 1)])
    masses = np.array([params["m{}".format(b)] for b in range(1, n_bodies + 1)])

    for body in range(1, n_bodies + 1):
        sma = params["sma{}".format(body)]
        mass = params["m{}".format(body)]
        mtot = params["m0"] + np.sum(masses[smas <= sma])
        ra, dec = kepler.calc_orbit(
            epochs_mjd,
            sma,
            params["ecc{}".format(body)],
            params["inc{}".format(body)],
            params["aop{}".format(body)],
            params["pan{}".format(body)],
            params["tau{}".format(body)],
            params["plx"],
            mtot,
            tau_ref_epoch=params["tau_ref_epoch"],
        )[:2]
        ra_star -= mass / mtot * np.atleast_1d(ra)
        dec_star -= mass / mtot * np.atleast_1d(dec)
    return ra_star, dec_star


def simulate_dr4(
    gost,
    params,
    *,
    g_mag=None,
    sigma_al=None,
    n_ccd=8,
    noise_floor=0.0,
    jitter=0.0,
    add_noise=True,
    seed=None,
    ref_epoch_mjd=DR4_REF_EPOCH_MJD,
    clip_to_dr4=True,
    source_id=0,
    include_components=False,
    output_file=None,
    overwrite=False,
):
    """Simulate transit-level Gaia DR4 along-scan astrometry for one star.

    Args:
        gost: Gaia scanning law for the target; see :func:`read_gost`.
        params (dict): injected truth, in orbitize's standard basis and units:

            - ``plx`` [mas] (required)
            - ``m0`` [Msun] (required if there are companions)
            - per companion ``i`` = 1, 2, ...: ``sma{i}`` [au], ``ecc{i}``,
              ``inc{i}``, ``aop{i}``, ``pan{i}`` [rad], ``tau{i}``,
              ``m{i}`` [Msun]. No companion keys simulates a single star.
            - ``alpha0``, ``delta0`` [mas], ``pm_ra``,
              ``pm_dec`` [mas/yr] (default 0); same names as the
              parameters ``DR4LogProb`` adds to the System.
            - ``tau_ref_epoch`` [MJD] (default 58849, the System default).

            Use :func:`params_from_sample` to build this from a posterior.
        g_mag (float): G magnitude, used to look up the per-transit uncertainty
            (see :func:`al_uncertainty`) when ``sigma_al`` is not given.
        sigma_al (float or array): per-transit along-scan uncertainty [mas];
            overrides ``g_mag``. An array must have one value per transit
            *after* any clipping to the DR4 time span.
        n_ccd (int), noise_floor (float): passed to :func:`al_uncertainty`.
        jitter (float): extra white noise [mas] added to the measurements but
            *not* to the reported uncertainties (to test a fitted jitter).
        add_noise (bool): False returns the noiseless model.
        seed (int or numpy.random.Generator): for reproducible noise.
        ref_epoch_mjd (float): reference epoch of the linear model, MJD(TCB).
        clip_to_dr4 (bool): drop transits outside the DR4 time span.
        source_id (int): written to the ``source_id`` column.
        include_components (bool): also return the noiseless pieces
            (``linear_al``, ``parallax_al``, ``orbit_al``, ``noise_al``) and the
            star's barycentric orbit (``ra_star``, ``dec_star``) as columns.
        output_file (str or pathlib.Path, optional): also save the table here.
            ``.ecsv`` (recommended) keeps units, MJD times and the injected
            truth. ``.csv`` writes JD(TCB) times, like a raw OHP/GOST file, and
            drops the metadata. Either can be passed to ``DR4LogProb``.
        overwrite (bool): allow replacing an existing ``output_file``.

    Returns:
        astropy.table.Table: one row per transit with ``source_id``,
        ``transit_id``, ``obs_time_tcb`` [MJD, TCB], ``centroid_pos_al`` [mas],
        ``centroid_pos_error_al`` [mas], ``parallax_factor_al``,
        ``scan_pos_angle`` [rad], ``field_of_view``. ``meta`` records the time
        format, reference epoch, noise settings and the injected ``truth``.
        Use ``output_file`` (or ``table.write("name.ecsv")``) to save it.
    """
    scans = read_gost(gost, clip_to_dr4=clip_to_dr4)
    truth, n_bodies = _parse_params(params)
    n_obs = len(scans)

    if sigma_al is None:
        if g_mag is None:
            raise ValueError("Provide either g_mag or sigma_al.")
        sigma = np.full(n_obs, al_uncertainty(g_mag, n_ccd, noise_floor))
    else:
        sigma = np.broadcast_to(np.asarray(sigma_al, dtype=float), (n_obs,)).copy()
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Along-scan uncertainties must be finite and positive.")
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and non-negative.")

    epochs = np.asarray(scans["obs_time_tcb"])
    sin_scan = np.sin(scans["scan_pos_angle"])
    cos_scan = np.cos(scans["scan_pos_angle"])
    dt_yr = (epochs - ref_epoch_mjd) / 365.25

    linear_al = (
        truth["alpha0"] * sin_scan
        + truth["delta0"] * cos_scan
        + truth["pm_ra"] * dt_yr * sin_scan
        + truth["pm_dec"] * dt_yr * cos_scan
    )
    parallax_al = truth["plx"] * np.asarray(scans["parallax_factor_al"])
    ra_star, dec_star = _star_orbit(epochs, truth, n_bodies)
    orbit_al = ORBIT_AL_SIGN * (ra_star * sin_scan + dec_star * cos_scan)

    if add_noise:
        rng = np.random.default_rng(seed)
        noise_al = rng.normal(0.0, np.hypot(sigma, jitter))
    else:
        noise_al = np.zeros(n_obs)

    out = Table()
    out["source_id"] = np.full(n_obs, source_id, dtype=np.int64)
    out["transit_id"] = np.arange(n_obs, dtype=np.int64)
    out["obs_time_tcb"] = epochs * u.day
    out["centroid_pos_al"] = (linear_al + parallax_al + orbit_al + noise_al) * u.mas
    out["centroid_pos_error_al"] = sigma * u.mas
    out["parallax_factor_al"] = (
        np.asarray(scans["parallax_factor_al"]) * u.dimensionless_unscaled
    )
    out["scan_pos_angle"] = np.asarray(scans["scan_pos_angle"]) * u.rad
    out["field_of_view"] = scans["field_of_view"]
    if include_components:
        for name, values in (
            ("linear_al", linear_al),
            ("parallax_al", parallax_al),
            ("orbit_al", orbit_al),
            ("noise_al", noise_al),
            ("ra_star", ra_star),
            ("dec_star", dec_star),
        ):
            out[name] = values * u.mas

    out.meta.update(
        time_format="mjd",
        time_scale="tcb",
        input_format="simulated",
        observation_level="transit",
        ref_epoch_mjd=float(ref_epoch_mjd),
        jitter=float(jitter),
        add_noise=bool(add_noise),
        seed=seed if isinstance(seed, (int, type(None))) else "generator",
        orbit_al_sign=ORBIT_AL_SIGN,
        truth=truth,
    )
    if output_file is not None:
        write_dr4(out, output_file, overwrite=overwrite)
    return out


def write_dr4(table, output_file, overwrite=False):
    """Save a simulated DR4 table so that ``DR4LogProb`` can read it back.

    ``.ecsv`` keeps everything. ``.csv`` stores ``obs_time_tcb`` as JD(TCB),
    which is what orbitize assumes for a plain CSV, and drops the metadata.
    """
    path = Path(output_file)
    suffix = path.suffix.lower()
    if suffix == ".ecsv":
        table.write(path, format="ascii.ecsv", overwrite=overwrite)
    elif suffix == ".csv":
        flat = table.copy()
        flat["obs_time_tcb"] = np.asarray(flat["obs_time_tcb"]) + 2400000.5
        flat.meta.clear()
        flat.write(path, format="ascii.csv", overwrite=overwrite)
    else:
        raise ValueError("output_file must end in .ecsv or .csv, not {!r}".format(path.name))
    return path


def params_from_sample(system, sample, **overrides):
    """Turn one posterior sample (e.g. from an RV fit) into a truth dict.

    Args:
        system (orbitize.system.System): the System the sample came from.
        sample (array): one row of ``results.post``, in the System's fitting
            basis (e.g. ``post[np.nanargmax(lnlike)]``).
        **overrides: values to inject instead, e.g. ``inc1=np.pi / 2``,
            ``pan1=0.3``, ``pm_ra=-974.8``. Parameters of the fitting basis
            are replaced *before* converting to the standard basis, so that
            derived quantities stay consistent (in the SemiAmp basis, m1 is
            recomputed from K1 and the injected inc1).

    Returns:
        dict: input for :func:`simulate_dr4`, including ``tau_ref_epoch``.
    """
    sample = np.array(sample, dtype=float)
    if sample.ndim != 1:
        raise ValueError("sample must be a single 1-D parameter vector.")

    for name in list(overrides):
        if name in system.param_idx:
            sample[system.param_idx[name]] = overrides.pop(name)

    standard = system.basis.to_standard_basis(sample.copy())
    idx = system.basis.standard_basis_idx

    params = {"plx": standard[idx["plx"]], "tau_ref_epoch": system.tau_ref_epoch}
    if "m0" not in idx:
        raise ValueError(
            "The System must fit companion masses (fit_secondary_mass=True) "
            "to predict the star's motion."
        )
    params["m0"] = standard[idx["m0"]]
    for body in range(1, system.num_secondary_bodies + 1):
        for name in _ORBIT_PARAMS:
            key = "{}{}".format(name, body)
            params[key] = standard[idx[key]]
    for name in _LINEAR_PARAMS:
        if name in system.param_idx:
            params[name] = sample[system.param_idx[name]]

    params.update(overrides)  # anything not in the fitting basis
    return {k: float(v) for k, v in params.items()}