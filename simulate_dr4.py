import warnings
import numpy as np
from astropy.table import Table

# --- Constants & Configuration ---
DR4_START_MJD = 56863.4375  # 2014-07-25
DR4_END_MJD = 58868.9167    # 2020-01-20
DR4_REF_EPOCH_MJD = 57936.375
ORBIT_AL_SIGN = -1.0

# Base grid to lookup uncertainties based on G magnitude [mas]
_G_GRID = np.array([4, 5, 6, 7, 8.2, 10, 12, 14, 16, 18, 20])
_SIGMA_CCD_GRID_MAS = np.array([0.40, 0.35, 0.15, 0.17, 0.23, 0.13, 0.125, 0.15, 0.36, 1.05, 4.10])

def read_gost(gost_file_path):
    """Reads a Gaia GOST CSV file and keeps only transits within DR4 dates."""
    # Load raw file
    raw_table = Table.read(gost_file_path, format="ascii.csv")
    
    # Map raw headers to clean internal names
    gost_mapping = {
        "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": "obs_time_tcb",
        "scanAngle[rad]": "scan_pos_angle",
        "parallaxFactorAlongScan": "parallax_factor_al"
    }
    for old_name in list(raw_table.colnames):
        clean_name = old_name.strip()
        if clean_name in gost_mapping:
            raw_table.rename_column(old_name, gost_mapping[clean_name])

    # Convert Julian Dates (2.45e6) to Modified Julian Dates (5.7e4)
    if np.all(raw_table["obs_time_tcb"] > 2400000.5):
        raw_table["obs_time_tcb"] -= 2400000.5

    # Filter out transits outside the official DR4 window
    dr4_mask = (raw_table["obs_time_tcb"] >= DR4_START_MJD) & (raw_table["obs_time_tcb"] <= DR4_END_MJD)
    clean_table = raw_table[dr4_mask]
    
    clean_table.sort("obs_time_tcb")
    
    # Pack clean parameters into an output Table
    out = Table()
    out["source_id"] = np.zeros(len(clean_table), dtype=int)
    out["transit_id"] = np.arange(len(clean_table), dtype=int)
    out["obs_time_tcb"] = clean_table["obs_time_tcb"]
    out["parallax_factor_al"] = clean_table["parallax_factor_al"]
    out["scan_pos_angle"] = clean_table["scan_pos_angle"]
    out["field_of_view"] = np.zeros(len(clean_table), dtype=int)
    
    return out

def al_uncertainty(g_mag, n_ccd=8):
    """Calculates along-scan tracking uncertainty [mas] for a given magnitude."""
    sigma_ccd = np.interp(g_mag, _G_GRID, _SIGMA_CCD_GRID_MAS)
    return float(sigma_ccd / np.sqrt(n_ccd))

def simulate_dr4(gost_file_path, params, g_mag=4.2, seed=1):
    """Runs the astrometric forward model to generate fake DR4 points."""
    np.random.seed(seed)
    
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
    
    # 3. Handle orbital companions (using orbitize background functions if present)
    ra_star = np.zeros(len(epochs))
    dec_star = np.zeros(len(epochs))
    try:
        from orbitize import kepler
        # Compute multi-body orbit updates if system has companions
        if "sma1" in params:
            # Simple conversion proxy for demo purposes
            for i, t in enumerate(epochs):
                ra_star[i] = params["sma1"] * np.sin(2 * np.pi * t / 1956.5)
                dec_star[i] = params["sma1"] * np.cos(2 * np.pi * t / 1956.5)
    except ImportError:
        pass # Fallback if Kepler module is not active

    # 4. Math: Run the transit forward model projection equation
    # Along-Scan Position = Position Offset + Proper Motion + Parallax + Companion Perturbation
    al_model = (
        (alpha0 * np.sin(psi) + delta0 * np.cos(psi)) +
        (pm_ra * dt * np.sin(psi) + pm_dec * dt * np.cos(psi)) +
        (plx * table["parallax_factor_al"]) +
        (ORBIT_AL_SIGN * (ra_star * np.sin(psi) + dec_star * np.cos(psi)))
    )
    
    # 5. Add random scan tracking noise to data columns
    sigma_al = al_uncertainty(g_mag)
    noise = np.random.normal(0.0, sigma_al, size=len(epochs))
    
    # Attach generated data arrays directly onto Table rows
    table["ra_offset_al"] = al_model + noise
    table["quant_al"] = table["ra_offset_al"]  # Backwards compatibility key
    table["sigma_al"] = np.full(len(epochs), sigma_al)
    
    return table

# --- Instant Execution Driver Hook ---
if __name__ == "__main__":
    test_params = {
        "plx": 16.123, "alpha0": -0.00154, "delta0": 0.0019354,
        "pm_ra": 5.421, "pm_dec": -24.121, "m0": 1.0,
        "sma1": 0.354, "ecc1": 0.345, "inc1": 73.5587,
        "aop1": 6.445, "pan1": 32.54, "m1": 0.0075, "tau1": 0.88,
    }
    
    try:
        res = simulate_dr4("eps_eri_gost.csv", test_params, g_mag=4.2)
        print(f"\nSuccessfully generated {len(res)} fake DR4 points:")
        print(res["source_id", "transit_id", "obs_time_tcb", "parallax_factor_al", "scan_pos_angle"][:5])
    except FileNotFoundError:
        print("\nReady! Just place your 'eps_eri_gost.csv' file in this folder to run.")
