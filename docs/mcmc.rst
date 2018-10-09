.. _mcmc-label:

MCMC Orbit Fitting
==================
Here, we will explain how to sample an orbit posterior using MCMC techniques. MCMC samplers take some time
to fully converge on the complex posterior, but should be able to explore all posteriors in roughly the same
amount of time (unlike OFTI). We will use the parallel-tempered version of the affine invariant sample from
the emcee package, as the parallel tempering helps the walkers get out of local minima.

Read in Data and Set up System
-------------------------------
We will read in the data using it is in the csv format defined by :py:class:`orbitize.read_input.read_input`.

.. code-block:: python

    from orbitize import read_input

    data_table = read_input.read_formatted_file("my+astrometry.csv")

Then we will set up the 2-body system. Orbitize can also fit for the total mass of the system and system parallax,
including marginalizing over the uncertainties in those parameters.

.. code-block:: python

    from orbitize import system

    # system parameters
    num_secondary_bodies = 1
    system_mass = 1.75 # Msol
    plx = 51.44 #mas
    mass_err = 0.05 # Msol
    plx_err = 0.12 #mas

    my_system = system.System(num_secondary_bodies, data_table, system_mass,
                              plx, mass_err=mass_err, plx_err=plx_err)


Setting up the MCMC Sampler
---------------------------
When setting up the sampler, we need to decide on how many temperatures and how many walkers per temperature
to use. Increasing the number of temperatures further ensures your walkers will explore all of parameter space
and will not get stuck in local minima. Increasing the number of walkers gives you more samples to use, and, for
the affine-invariant sampler, a minimum amount is required for good convergence. Of course, the trade off is that
more samplers means more computation time. Here, we recommend 20 temperatures and 1000 walkers, as we find this
to be reliable in always converging. Note that we will only use the samples from the lowest temperature walkers only.
We also will assume that our astrometric measurements follow a Gaussian distribution.

.. code-block:: python

    import multiprocessing as mp
    from orbitize import sampler

    # Sampler parameters
    likelihood_func_name='chi2_lnlike'
    n_temps = 20
    n_walkers = 1000
    n_threads = mp.cpu_count() # or a different number if you prefer

    my_sampler = sampler.PTMCMC(likelihood_func_name, my_system, n_temps, n_walkers, n_threads)


Running the MCMC Sampler
------------------------
We need to pick how many steps the MCMC sampler should sample. Additionally, because the samples are correlated,
we often only save every nth sample. This helps when we run a lot of samples, and saving all the samples requires
too much disk space, despite the fact many samples are unnecessary because they are correlated.

.. code-block:: python

    total_orbits = 10000000 # number of steps x number of walkers (at lowest temperature)
    burn_steps = 1000 # steps to burn in per walker
    thin = 10 # only save every 10th steps

    my_sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)


After completing the samples, the `'run_sampler'` method also creates a `'Results'` object that can be accessed
with `'my_sampler.results'`.

Plotting Basics
---------------
We will make some basic plots to visualize the samples in `'my_sampler.results'`. orbitize currently has two basic
plotting functions which returns matplotlib Figure objects. First, we can make a corner plot (also known as
triangle plot, scatterplot matrix, pairs plot) to visualize correlations between pairs of orbit parameters:

.. code-block: python

    corner_plot_fig = my_sampler.results.plot_corner() # Creates a corner plot and returns Figure object
    corner_plot_fig.savefig('my_corner_plot.png') # This is matplotlib.figure.Figure.savefig()


Next, we can plot a visualization of a selection of orbits sampled by our sampler. By default, the first epoch
plotted is the year 2000 and 100 sampled orbits are displayed.

.. code-block: python
    orbit_plot_fig = my_sampler.results.plot_orbits(
                        object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
                        num_orbits_to_plot= 100 # Will plot 100 randomly selected orbits of this companion
                        )
    orbit_plot_fig.savefig('my_orbit_plot.png') # This is matplotlib.figure.Figure.savefig()


For more advanced plotting options and suggestions on what to do with the returned matplotlib Figure objects,
see the dedicated Plotting tutorial (coming soon).


Saving and Loading Results
--------------------------
We will save the results in the HDF5 format. It will save two datasets: `'post'` which will contain the posterior
(the chains of the lowest temperature walkers) and `'lnlike'` which has the corresponding probabilities. In addition,
it saves `'sampler_name'` as an attribute of the HDF5 root group.

.. code-block:: python

    my_sampler.results.save_result("my_posterior.hdf5")


Saving sampler results is a good idea when we want to analyze the results in a different script or when we you want to
save the output of a long MCMC run to avoid having to re-run it in the future. We can then load the saved results into
a new blank results object.

.. code-block: python

    from orbitize import results
    loaded_results = results.Results() # Create blank results object for loading
    loaded_results.load_results("my_posterior.hdf5")


Instead of loading results into an orbitize.results.Results object, we can also directly access the saved data using
the `'h5py'` python module

.. code-block: python

      import h5py
      filename = 'my_posterior.hdf5'
      hf = h5py.File(filename,'r') # Opens file for reading
      # Load up each dataset from hdf5 file
      sampler_name = np.str(hf.attrs['sampler_name'])
      post = np.array(hf.get('post'))
      lnlike = np.array(hf.get('lnlike'))


Although HDF5 is the recommend and default way to save results, we can also save and load as a Binary FITS table.

.. code-block: python

    # Saving results object
    my_sampler.results.save_result("my_posterior.fits", format='fits')

    # Loading results object
    from orbitize import results
    loaded_results = results.Results() # Create blank results object for loading
    loaded_results.load_results("my_posterior.fits", format='fits')


Test.
