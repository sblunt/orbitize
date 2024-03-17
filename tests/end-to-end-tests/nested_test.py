import orbitize
from orbitize import read_input, system, sampler, priors
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot
import time


savedir = "."

"""
Runs the GJ504 fit (from the quickstart tutorial) using dynesty as a backend

Written: Thea McKenna, 2023
"""


def dynesty_e2e_test():

    data_table = read_input.read_file("{}/GJ504.csv".format(orbitize.DATADIR))

    # number of secondary bodies in system
    num_planets = 1

    # total mass & error [msol]
    total_mass = 1.22
    mass_err = 0  # 0.08

    # parallax & error[mas]
    plx = 56.95
    plx_err = 0  # 0.26

    sys = system.System(
        num_planets,
        data_table,
        total_mass,
        plx,
        mass_err=mass_err,
        plx_err=plx_err,
        restrict_angle_ranges=True,
    )
    # alias for convenience
    lab = sys.param_idx

    # set prior on semimajor axis
    sys.sys_priors[lab["sma1"]] = priors.LogUniformPrior(10, 300)

    nested_sampler = sampler.NestedSampler(sys)

    start = time.time()

    samples, num_iter = nested_sampler.run_sampler(num_threads=50)
    nested_sampler.results.save_results("{}/nested_sampler_test.hdf5".format(savedir))
    print("iteration number is: " + str(num_iter))

    print("iteration time: {:.f} mins".format((time.time() - start) / 60.0))

    fig, ax = plt.subplots(2, 1)
    accepted_eccentricities = nested_sampler.results.post[:, lab["ecc1"]]
    accepted_inclinations = nested_sampler.results.post[:, lab["inc1"]]
    ax[0].hist(accepted_eccentricities, bins=50)
    ax[1].hist(accepted_inclinations, bins=50)
    ax[0].set_xlabel("ecc")
    ax[1].set_xlabel("inc")
    plt.tight_layout()
    plt.savefig("{}/nested_sampler_test.png".format(savedir))

    fig, axes = dyplot.traceplot(nested_sampler.dynesty_sampler.results)
    plt.savefig("{}/nested_sampler_traceplot.png".format(savedir))


if __name__ == "__main__":
    dynesty_e2e_test()
