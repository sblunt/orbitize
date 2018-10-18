.. _mcmc-label:

MCMC Orbit Fitting
==================
Here, we will explain how to sample an orbit posterior using MCMC techniques. MCMC samplers take some time
to fully converge on the complex posterior, but should be able to explore all posteriors in roughly the same
amount of time (unlike OFTI). We will use the parallel-tempered version of the affine invariant sample from
the emcee package, as the parallel tempering helps the walkers get out of local minima. Parallel-tempering can
be disabled by setting the number of temperatures to 1, and will revert back to using the regular ensemble 
sampler from emcee. 

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
more samplers means more computation time. Here, we recommend 20 temperatuers and 1000 walkers, as we find this 
to be reliable in always converging. Note that we will only use the samples from the lowest temperature walkers only.
We also will assume that our astrometric measuremnts follow a Gaussian distribution. 

.. code-block:: python

    import multiprocessing as mp
    from orbitize import sampler

    # Sampler parameters
    likelihood_func_name='chi2_lnlike'
    n_temps = 20
    n_walkers = 1000
    n_threads = mp.cpu_count() # or a different number if you prefer

    my_sampler = sampler.MCMC(likelihood_func_name, my_system, n_temps, n_walkers, n_threads)


Running the MCMC Sampler
------------------------
We need to pick how many steps the MCMC sampler should sample. Additionally, because the samples are correalted,
we often only save every nth sample. This helps when we run a lot of samples, and saving all the samples requires
too much disk space, despite the fact many samples are unncessary because they are correlated. 

.. code-block:: python

    total_orbits = 10000000 # number of steps x number of walkers (at lowest temperature)
    burn_steps = 1000 # steps to burn in per walker
    thin = 10 # only save every 10th steps

    my_sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)


Save Results
------------
We will save the results in the HDF5 format. It will save two fields: `'post'` which will contain the posterior 
(the chains of the lowest temperature walkers) and `'lnlike'` which has the corresponding probabilities.

.. code-block:: python

    my_sampler.results.save_result("my_posterior.hdf5")