from orbitize import driver, system, basis, DATADIR
import numpy as np

basis_names = {'Standard' : ['sma', 'ecc', 'inc', 'aop', 'pan', 'tau'], 
	'Period' : ['per', 'ecc', 'inc', 'aop', 'pan', 'tau'], 
	'SemiAmp': ['per', 'ecc', 'inc', 'aop', 'pan', 'tau', 'K'] , 
	'XYZ' : ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']}

def test_no_extra_data():
	filename = "{}/GJ504.csv".format(DATADIR)

	num_secondary_bodies = 1
	system_mass = 1.75 # [Msol]
	plx = 51.44 # [mas]
	mass_err = 0.05 # [Msol]
	plx_err = 0.12 # [mas]

	# System Mass (fit secondary mass is false)
	for basis in basis_names:
		if basis != 'SemiAmp':
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'mtot']
		else:
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'm0']			

		my_driver = driver.Driver(
		    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
		    fitting_basis=basis
		)

		assert expected_labels == my_driver.system.labels

	# Single Companion
	for basis in basis_names:
		if basis != 'SemiAmp':
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'm1', 'm0']
		else:
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'm0']			

		my_driver = driver.Driver(
		    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
		    fitting_basis=basis, system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0}
		)

		assert expected_labels == my_driver.system.labels	

	# Multiple Companions

def test_with_rv():
	filename = "{}/HD4747.csv".format(DATADIR)

	num_secondary_bodies = 1
	system_mass = 0.84 # [Msol]
	plx = 53.18 # [mas]
	mass_err = 0.04 # [Msol]
	plx_err = 0.12 # [mas]

	# Single Body
	for basis in basis_names:
		if basis != 'SemiAmp':
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'gamma_defrv', 'sigma_defrv', 'm1', 'm0']
		else:
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'gamma_defrv', 'sigma_defrv', 'm0']

		my_driver = driver.Driver(
		    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
		    fitting_basis=basis, system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0}
		)

		assert expected_labels == my_driver.system.labels

def test_with_hip_iad():
	filename = "{}/betaPic.csv".format(DATADIR)
	hipp_filename = "{}/HIP027321.d".format(DATADIR)

	num_secondary_bodies = 1
	system_mass = 1.75
	mass_err = 0
	plx = 51.44
	plx_err = 0.12

	hip_labels = ['pm_ra', 'pm_dec', 'alpha0', 'delta0']

	for basis in basis_names:
		if basis != 'SemiAmp':
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx'] + hip_labels + ['m1', 'm0']
		else:
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx'] + hip_labels + ['m0']

		my_driver = driver.Driver(
			filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
			fitting_basis = basis, system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0, 
			'hipparcos_number':27321, 'hipparcos_filename':hipp_filename}
		)

		assert expected_labels == my_driver.system.labels

if __name__ == '__main__':
	test_no_extra_data()
	test_with_rv()
	test_with_hip_iad()