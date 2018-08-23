"""
Test the routines in the orbitize.Results module
"""
# Based on driver.py

from orbitize import read_input, system, sampler, priors, results
import matplotlib.pyplot as plt

def create_test_system_object():
    """
    Returns a system object, based on beta pic, to test Results module
    """
    # System parameters
    datafile='test_bpic_val.csv'
    num_secondary_bodies=1
    system_mass=1.75 # Msol
    plx=51.44 #mas
    mass_err=0.05 # Msol
    plx_err=0.12 #mas

    # Read in data
    data_table = read_input.read_formatted_file(datafile)

    # Initialize System object which stores data & sets priors
    test_system = system.System(
        num_secondary_bodies, data_table, system_mass,
        plx, mass_err=mass_err, plx_err=plx_err
    )

    # We could overwrite any priors we want to here.
    # Using defaults for now.

    return test_system


def create_and_run_test_sampler_object(test_system):
    """
    Returns a PTsampler object, based on beta pic, to test Results module.
    Uses system object as input
    """
    # Sampler parameters, just to test Results module
    likelihood_func_name='chi2_lnlike'
    n_temps=1
    n_walkers=20
    n_threads=2
    total_orbits=100 # n_walkers x num_steps_per_walker
    burn_steps=1

    # Initialize Sampler object, which stores information about
    # the likelihood function & the algorithm used to generate
    # orbits, and has System object as an attribute.
    test_sampler = sampler.PTMCMC(likelihood_func_name,test_system,n_temps,n_walkers,n_threads)

    # Run the sampler to compute some orbits, yeah!
    # This also creates and stores a Results object in test_sampler.Results
    test_sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=1)

    return test_sampler


def test_plot_corner(results):
    Figure = results.plot_corner()
    return Figure


def test_add_orbits():
    """
    Tests adding orbits to Results object
    """
    pass

if __name__ == "__main__":
    # Initialize system object
    system = create_test_system_object()
    # Create and run sampler object (which creates Results object)
    sampler = create_and_run_test_sampler_object(system)
    # Store results object in system.results
    system.add_results(sampler.results)
    # Run some tests results.Results object stored in sampler.results
    corner_fig = test_plot_corner(system.results[0])
    plt.show()
    import pdb; pdb.set_trace()
