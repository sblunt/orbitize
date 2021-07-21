'''
Make sure orbit plotting can still occur after chopping chains.
'''
import orbitize
from orbitize import driver, DATADIR
import multiprocessing as mp

def verify_results_data(res, sys):
	# Make sure data attribute in results class is not 'None'
	assert res.data is not None

	# Make sure the data tables are equivalent between classes
	res_data = res.data.to_pandas()
	sys_data = sys.data_table.to_pandas()
	assert res_data.equals(sys_data) == True

	# Make sure no error results when making the final orbit plot
	try:
		epochs = sys.data_table['epoch']
		res.plot_orbits(
			object_to_plot = 1,
			num_orbits_to_plot = 10,
			start_mjd = epochs[0]
		)
	except:
		raise Exception("Plotting orbits failed.")

def run_mcmc():
	'''
	First run MCMC sampler to generate results object and make a call to 'chop_chains'
	function afterwards.
	'''

	filename = "{}/HD4747.csv".format(DATADIR)

	num_secondary_bodies = 1
	system_mass = 0.84
	plx = 53.18
	mass_err = 0.04
	plx_err = 0.12

	num_temps = 5
	num_walkers = 40
	num_threads = mp.cpu_count()

	my_driver = driver.Driver(
		filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
		system_kwargs={'fit_secondary_mass':True, 'tau_ref_epoch':0},
		mcmc_kwargs={'num_temps':num_temps, 'num_walkers':num_walkers, 'num_threads':num_threads})

	mcmc_sys = my_driver.system

	total_orbits = 10000
	burn_steps = 10
	thin = 2

	my_driver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
	my_driver.sampler.chop_chains(burn=25, trim=25)

	mcmc_result = my_driver.sampler.results

	verify_results_data(mcmc_result, mcmc_sys)

if __name__ == '__main__':
	run_mcmc()

