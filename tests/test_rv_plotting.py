import orbitize
from orbitize import driver, DATADIR
import multiprocessing as mp

def test_rv_default_inst():
	# Initialize Driver to Run MCMC
	filename = '{}/HD4747.csv'.format(DATADIR)

	num_secondary_bodies = 1
	system_mass = 0.84 # [Msol]
	plx = 53.18 # [mas]
	mass_err = 0.04 # [Msol]
	plx_err = 0.12 # [mas]

	num_temps = 5
	num_walkers = 30
	num_threads = mp.cpu_count() # or a different number if you prefer

	my_driver = driver.Driver(
	    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
	    system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0},
	    mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads}
	)

	total_orbits = 1000
	burn_steps = 10
	thin = 2

	# Run Quick Sampler
	m = my_driver.sampler
	m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
	epochs = my_driver.system.data_table['epoch']

	# Test plotting with single orbit
	orbit_plot_fig = m.results.plot_orbits(
	    object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
	    num_orbits_to_plot= 1, # Will plot 100 randomly selected orbits of this companion
	    start_mjd=epochs[3], # Minimum MJD for colorbar (here we choose first data epoch)
	    rv_time_series = True,
	    plot_astrometry_insts = True
	)

	# Test plotting with multiple orbits
	orbit_plot_fig = m.results.plot_orbits(
	    object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
	    num_orbits_to_plot= 10, # Will plot 100 randomly selected orbits of this companion
	    start_mjd=epochs[3], # Minimum MJD for colorbar (here we choose first data epoch)
	    rv_time_series = True,
	    plot_astrometry_insts = True
	)

def test_rv_multiple_inst():
	filename = '{}/HR7672_joint.csv'.format(DATADIR)

	num_secondary_bodies = 1
	system_mass = 1.08 # [Msol]
	plx = 56.2 # [mas]
	mass_err = 0.04 # [Msol]
	plx_err = 0.01 # [mas]

	# MCMC parameters
	num_temps = 5
	num_walkers = 30
	num_threads = 2

	my_driver = driver.Driver(
	    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
	    system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0},
	    mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads}
	)

	total_orbits = 500
	burn_steps = 10
	thin = 2

	m = my_driver.sampler
	m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
	epochs = my_driver.system.data_table['epoch']

	orbit_plot_fig = m.results.plot_orbits(
	    object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
	    num_orbits_to_plot= 1, # Will plot 100 randomly selected orbits of this companion
	    start_mjd=epochs[0], # Minimum MJD for colorbar (here we choose first data epoch)
	    rv_time_series = True,
	    plot_astrometry_insts = True
	)

	# Test plotting with multiple orbits
	orbit_plot_fig = m.results.plot_orbits(
	    object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
	    num_orbits_to_plot= 10, # Will plot 100 randomly selected orbits of this companion
	    start_mjd=epochs[0], # Minimum MJD for colorbar (here we choose first data epoch)
	    rv_time_series = True,
	    plot_astrometry_insts = True
	)

if __name__ == '__main__':
	test_rv_default_inst()
	test_rv_multiple_inst()