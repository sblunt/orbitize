import numpy as np
from astropy.table import Table
from astropy import units as u
from orbitize import kepler

# --- Constants & Configuration ---
DR4_START_MJD = 56863.4375  # 2014-07-25
DR4_END_MJD = 58868.9167  # 2020-01-20
DR4_REF_EPOCH_MJD = 57936.375

# Approximate per-CCD along scan uncertainties [mas].
# Tabulation from Gaiamock, based on the "EDR3 adjusted" curve in
# Holl et al. (2023), Fig. 3:
# https://arxiv.org/abs/2206.05439
# https://github.com/kareemelbadry/gaiamock/blob/main/gaiamock.py
_G_GRID = np.array(
    [
        4,
        5,
        6,
        7,
        8.2,
        8.4,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ]
)
_SIGMA_CCD_GRID_MAS = np.array(
    [
        0.40,
        0.35,
        0.15,
        0.17,
        0.23,
        0.13,
        0.13,
        0.135,
        0.125,
        0.13,
        0.15,
        0.23,
        0.36,
        0.63,
        1.05,
        2.05,
        4.10,
    ]
)


def read_gost(gost_file_path):
    """Reads a Gaia GOST CSV file and keeps only transits within DR4 dates."""
    # Load raw file
    raw_table = Table.read(gost_file_path, format="ascii.csv")

    # Map raw headers to clean internal names
    gost_mapping = {
        "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": "obs_time_tcb",
        "scanAngle[rad]": "scan_pos_angle",
        "parallaxFactorAlongScan": "parallax_factor_al",
        "parallaxFactorAcrossScan": "parallax_factor_ac",
        "Fov[FovP=preceding/FovF=following]": "field_of_view",
    }
    for old_name in list(raw_table.colnames):
        clean_name = old_name.strip()
        if clean_name in gost_mapping:
            raw_table.rename_column(old_name, gost_mapping[clean_name])

    # Convert Julian Dates (2.45e6) to Modified Julian Dates (5.7e4)
    if np.all(raw_table["obs_time_tcb"] > 2400000.5):
        raw_table["obs_time_tcb"] -= 2400000.5

    # Filter out transits outside the official DR4 window
    dr4_mask = (raw_table["obs_time_tcb"] >= DR4_START_MJD) & (
        raw_table["obs_time_tcb"] <= DR4_END_MJD
    )
    clean_table = raw_table[dr4_mask]

    clean_table.sort("obs_time_tcb")

    # Pack clean parameters into an output Table
    out = Table()
    out["source_id"] = np.zeros(len(clean_table), dtype=int)
    out["transit_id"] = np.arange(len(clean_table), dtype=int)
    out["obs_time_tcb"] = clean_table["obs_time_tcb"]
    out["parallax_factor_al"] = clean_table["parallax_factor_al"]
    if "parallax_factor_ac" in clean_table.colnames:
        out["parallax_factor_ac"] = clean_table["parallax_factor_ac"]
        out["parallax_factor_ac"].unit = u.dimensionless_unscaled
    out["scan_pos_angle"] = clean_table["scan_pos_angle"]
    out["field_of_view"] = (
        clean_table["field_of_view"]
        if "field_of_view" in clean_table.colnames
        else np.zeros(len(clean_table), dtype=int)
    )
    return out


def al_uncertainty(g_mag, n_ccd=8):
    """Calculates along-scan tracking uncertainty [mas] for a given magnitude."""
    sigma_ccd = np.interp(g_mag, _G_GRID, _SIGMA_CCD_GRID_MAS)
    return float(sigma_ccd / np.sqrt(n_ccd))


def simulate_dr4(
    gost_file_path,
    params,
    g_mag=4.2,
    seed=1,
    tau_ref_epoch=58849,
):
    """Runs the astrometric forward model to generate fake DR4 points."""
    # 1. Load your observation timeline
    table = read_gost(gost_file_path)
    epochs = table["obs_time_tcb"]
    psi = table["scan_pos_angle"]

    # Time diff in years relative to the reference catalog epoch
    dt = (epochs - DR4_REF_EPOCH_MJD) / 365.25

    # 2. Extract background stellar movement paths
    alpha0 = params.get("alpha0", 0.0)
    delta0 = params.get("delta0", 0.0)
    pm_ra = params.get("pm_ra", 0.0)
    pm_dec = params.get("pm_dec", 0.0)
    plx = params.get("plx", 0.0)

    # 3. Sum the stellar reflex contributions from all supplied companions.
    ra_star = np.zeros(len(epochs))
    dec_star = np.zeros(len(epochs))

    bodies = sorted(int(name[3:]) for name in params if name.startswith("sma"))
    for body in bodies:
        # Match System's enclosed-mass convention for each orbit.
        mtot = params["m0"] + sum(
            params[f"m{other}"]
            for other in bodies
            if params[f"sma{other}"] <= params[f"sma{body}"]
        )
        ra_relative, dec_relative, *_ = kepler.calc_orbit(
            np.asarray(epochs, dtype=float),
            params[f"sma{body}"],
            params[f"ecc{body}"],
            params[f"inc{body}"],
            params[f"aop{body}"],
            params[f"pan{body}"],
            params[f"tau{body}"],
            plx,
            mtot,
            tau_ref_epoch=tau_ref_epoch,
        )
        reflex_scale = -params[f"m{body}"] / mtot
        ra_star += reflex_scale * np.asarray(ra_relative).reshape(-1)
        dec_star += reflex_scale * np.asarray(dec_relative).reshape(-1)

    # 4. Math: Run the transit forward model projection equation
    # Along-Scan Position = Position Offset + Proper Motion + Parallax + Companion Perturbation
    al_model = (
        (alpha0 * np.sin(psi) + delta0 * np.cos(psi))
        + (pm_ra * dt * np.sin(psi) + pm_dec * dt * np.cos(psi))
        + (plx * table["parallax_factor_al"])
        + ra_star * np.sin(psi)
        + dec_star * np.cos(psi)
    )

    # 5. Add random scan tracking noise to data columns
    sigma_al = al_uncertainty(g_mag)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma_al, size=len(epochs))

    # Attach generated data arrays directly onto Table rows
    table["centroid_pos_al"] = al_model + noise
    table["centroid_pos_error_al"] = np.full(len(epochs), sigma_al)

    # Keep the original names as aliases.
    table["ra_offset_al"] = table["centroid_pos_al"]
    table["quant_al"] = table["centroid_pos_al"]
    table["sigma_al"] = table["centroid_pos_error_al"]

    table["obs_time_tcb"].unit = u.day
    table["centroid_pos_al"].unit = u.mas
    table["centroid_pos_error_al"].unit = u.mas
    table["parallax_factor_al"].unit = u.dimensionless_unscaled
    table["scan_pos_angle"].unit = u.rad

    table.meta.update(
        time_format="mjd",
        time_scale="tcb",
        input_format="simulation",
        observation_level="transit",
    )

    return table
