"""
Stellar RVs + astrometry, using data for HD 4747.
"""

# Initialize Driver to Run MCMC
filename = "{}/HD4747.csv".format(DATADIR)

num_secondary_bodies = 1
system_mass = 0.84  # [Msol]
plx = 53.18  # [mas]
mass_err = 0.04  # [Msol]
plx_err = 0.12  # [mas]

num_temps = 5
num_walkers = 50
num_threads = 50  # or a different number if you prefer

my_driver = driver.Driver(
    filename,
    "MCMC",
    num_secondary_bodies,
    system_mass,
    plx,
    mass_err=mass_err,
    plx_err=plx_err,
    system_kwargs={"fit_secondary_mass": True, "tau_ref_epoch": 0},
    mcmc_kwargs={
        "num_temps": num_temps,
        "num_walkers": num_walkers,
        "num_threads": num_threads,
    },
)

total_orbits = 100_000
burn_steps = 1000
thin = 2

# Run Sampler
m = my_driver.sampler
m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)