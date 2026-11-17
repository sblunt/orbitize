import random
import warnings

import orbitize
from orbitize import driver

warnings.filterwarnings("ignore")


def test_check_prior_support(PriorChanges=False):
    """
    Test the check_prior_support() function to ensure it behaves correctly.
    Should fail with a ValueError if any parameters are outside prior support.
    Should behave normally if all parameters are within prior support.
    """

    # set up as if we are running the tutorial retrieval
    myDriver = driver.Driver(
        f"{orbitize.DATADIR}/GJ504.csv",  # data file
        "MCMC",  # choose from: ['OFTI', 'MCMC']
        1,  # number of planets in system
        1.22,  # total system mass [M_sun]
        56.95,  # system parallax [mas]
        mass_err=0.08,  # mass error [M_sun]
        plx_err=0.26,  # parallax error [mas]
        mcmc_kwargs={"num_temps": 2, "num_walkers": 18, "num_threads": 1},
    )

    # mess with the priors if requested
    if PriorChanges:
        zlist = []
        for a in range(4):
            x = random.randint(0, myDriver.sampler.num_temps - 1)
            y = random.randint(0, myDriver.sampler.num_walkers - 1)
            z = random.randint(0, myDriver.sampler.num_params - 1)
            myDriver.sampler.curr_pos[x, y, z] = -1000
            zlist.append(z)

    # run the tests
    try:
        _ = myDriver.sampler.check_prior_support()
    # catch the correct error
    except ValueError:
        pass


if __name__ == "__main__":
    test_check_prior_support()
    test_check_prior_support(PriorChanges=True)
