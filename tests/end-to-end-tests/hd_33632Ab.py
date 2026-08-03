"""
Compare to Hsu+ 2024 (case with HGCA, no RVs, and relative astrometry)

https://iopscience.iop.org/article/10.3847/1538-4357/ad58d3#apjad58d3t5
"""

import os
from orbitize import DATADIR, hipparcos, gaia, read_input, system, priors, sampler, results
import matplotlib.pyplot as plt

# the necessary input data for beta Pic is part of the orbitize! example data!
iad_filepath = os.path.join(DATADIR, "H024332.d")
gost_filepath = os.path.join(DATADIR, "gost_22.4.3_806543_2026-07-28-17-46-26_HD_33632.csv")

# Create the HGCA and helper Hipparcos object
hipparcos_lnprob = hipparcos.HipparcosLogProb(iad_filepath, 24332, 1)
hgca_lnprob = gaia.HGCALogProb(24332, hipparcos_lnprob, gost_filepath)

# read in relative astrometry
astrometry_filepath = os.path.join(DATADIR, "HD33632Ab.csv")
data_table = read_input.read_file(astrometry_filepath)

# set up the system, passing in hgca_lnprob and setting it fit dynamical mass
stellar_mass = 1.11
stellar_mass_err = 0.09
plx = 37.8953
plx_err = 0.0263

this_system = system.System(
    1,
    data_table,
    stellar_mass,
    plx,
    mass_err=stellar_mass_err,
    plx_err=plx_err,
    fit_secondary_mass=True,
    gaia=hgca_lnprob,
)

# adjust the prior on mass to be uniform between 0 and 0.1 Msol
# this_system.sys_priors[this_system.param_idx["m1"]] = priors.LogUniformPrior(
#     0, 0.1
# )

# MCMC parameters
n_temps=20
n_walkers=1000
n_threads=20
total_orbits= n_walkers * 50_000
burn_steps=10_000
thin=10

run_fit = True

if __name__ == '__main__':

    # create the sampler, run it, and save posteriors
    this_sampler = sampler.MCMC(this_system, n_temps, n_walkers, n_threads)

    output_filename = "HD_33632_Ab.hdf5"
    periodic_save_freq = 5_000

    if run_fit:

        this_sampler.run_sampler(
            total_orbits, burn_steps=burn_steps, thin=thin, periodic_save_freq=periodic_save_freq,
            output_filename=output_filename
        )

        this_sampler.results.save_results(output_filename)

    myResults = results.Results()
    myResults.load_results(output_filename)

    # make corner plot
    fig = myResults.plot_corner()
    plt.savefig("HD_33632_Ab.png", dpi=250)