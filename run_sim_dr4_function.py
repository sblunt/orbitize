import simulate_dr4

# 1. Map your parameters from target_1.csv data (Now including tau1 and m1)
params = {
    "plx": 16.123,               # parallax_mas
    "alpha0": -0.00154,          # alpha_mas
    "delta0": 0.0019354,         # delta_mas
    "pm_ra": 5.421,              # mu_alpha_mas_yr
    "pm_dec": -24.121,           # mu_delta_mas_yr
    "m0": 1.0,                   # Primary mass (in M_sun)
    
    # Companion 1 Parameters
    "sma1": 0.354,               # a0_mas (semi-major axis)
    "ecc1": 0.345,               # e (eccentricity)
    "inc1": 73.5587,             # i_deg (inclination)
    "aop1": 6.445,               # omega_deg (argument of periastron)
    "pan1": 32.54,               # big_omega_deg (position angle of nodes)
    "m1": 0.0075,                # Derived companion mass in Solar Masses (~7.8 Jupiter masses)
    "tau1": 0.88,                # Derived periastron passage fraction (0.0 to 1.0 range)
}

# 2. Read your Epsilon Eridani scanning law file
scanning_law = simulate_dr4.read_gost("/Users/astrovidee/orbitize/orbitize/example_data/eps_eri_gost.csv")

# 3. Simulate the DR4 transits 
table = simulate_dr4.simulate_dr4("/Users/astrovidee/orbitize/orbitize/example_data/eps_eri_gost.csv", params, g_mag=4.2, seed=1)

print("Simulation Complete!")
print(table)
