from orbitize import driver, system, basis, DATADIR, read_input, hipparcos
import numpy as np
import pytest

basis_names = {'Standard' : ['sma', 'ecc', 'inc', 'aop', 'pan', 'tau'], 
	'Period' : ['per', 'ecc', 'inc', 'aop', 'pan', 'tau'], 
	'SemiAmp': ['per', 'ecc', 'inc', 'aop', 'pan', 'tau', 'K'], 
	'XYZ' : ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']}

def test_no_extra_data():
	"""
	Make sure that the labels are generated properly for all of the basis sets 
	for configurations of the driver where (1) system mass, (2) single companion 
	mass, and (3) two companion masses are being fitted. In any case, RV or 
	Hipparcos parameters are not being fitted.

	For XYZ, expect there to be exceptions thrown when making the driver.
	"""
	filename = "{}/GJ504.csv".format(DATADIR)

	num_secondary_bodies = 1
	system_mass = 1.75 # [Msol]
	plx = 51.44 # [mas]
	mass_err = 0.05 # [Msol]
	plx_err = 0.12 # [mas]

	# (1) System Mass (fit secondary mass is false)
	for basis in basis_names:
		if basis != 'SemiAmp':
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'mtot']
		else:
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'm0']			

		if basis == 'XYZ': # Should throw error for XYZ basis
			with pytest.raises(Exception) as excinfo:
				my_driver = driver.Driver(
				    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
				    system_kwargs = {'fitting_basis': basis}
				)
			assert str(excinfo.value) == "For now, the XYZ basis requires data in RA and DEC offsets."
		else:
			my_driver = driver.Driver(
			    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
			    system_kwargs = {'fitting_basis': basis}
			)

			assert expected_labels == my_driver.system.labels

	# (2) Single Companion
	for basis in basis_names:
		if basis != 'SemiAmp':
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'm1', 'm0']
		else:
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'm0']

		if basis == 'XYZ': # Should throw error for XYZ basis
			with pytest.raises(Exception) as excinfo:
				my_driver = driver.Driver(
				    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
				    system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0, 'fitting_basis':basis}
				)
			assert str(excinfo.value) == "For now, the XYZ basis requires data in RA and DEC offsets."

		else:
			my_driver = driver.Driver(
			    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
			    system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0, 'fitting_basis':basis}
			)

			assert expected_labels == my_driver.system.labels	

	# (3) Multiple Companions
	filename = "{}/test_val_multi.csv".format(DATADIR)
	for basis in basis_names:
		init_labels = [item + '1' for item in basis_names[basis]] + [item + '2' for item in basis_names[basis]]
		if basis != 'SemiAmp':
			expected_labels = init_labels + ['plx', 'm1', 'm2', 'm0']
		else:
			expected_labels = init_labels + ['plx', 'm0']

		if basis == 'XYZ': # Should throw error for XYZ basis
			with pytest.raises(Exception) as excinfo:
				my_driver = driver.Driver(filename, 'MCMC', 2, 1.52, 24.76, mass_err=0.15, plx_err=0.64,
					system_kwargs={'fit_secondary_mass':True, 'tau_ref_epoch':True, 'fitting_basis':basis}
				)
			assert str(excinfo.value) == "For now, the epoch with the lowest sepparation error should not be one of the last two entries for body1"
		else:
			my_driver = driver.Driver(filename, 'MCMC', 2, 1.52, 24.76, mass_err=0.15, plx_err=0.64,
				system_kwargs={'fit_secondary_mass':True, 'tau_ref_epoch':True, 'fitting_basis':basis}
			)
			assert expected_labels == my_driver.system.labels

def test_with_rv():
	'''
	Make sure the labels are generated correctly for all of the basis sets for the configuration of
	the driver where RV data is supplied. Again, for the XYZ basis, expect there to be errors thrown.
	'''
	filename = "{}/HD4747.csv".format(DATADIR)

	num_secondary_bodies = 1
	system_mass = 0.84 # [Msol]
	plx = 53.18 # [mas]
	mass_err = 0.04 # [Msol]
	plx_err = 0.12 # [mas]

	# (1) Single Body
	for basis in basis_names:
		if basis != 'SemiAmp':
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'gamma_defrv', 'sigma_defrv', 'm1', 'm0']
		else:
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx', 'gamma_defrv', 'sigma_defrv', 'm0']

		if basis == 'XYZ':
			with pytest.raises(Exception) as excinfo:
				my_driver = driver.Driver(
				    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
				    system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0, 'fitting_basis':basis}
				)
			assert str(excinfo.value) == "For now, the XYZ basis requires data in RA and DEC offsets."
		else:
			my_driver = driver.Driver(
			    filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
			    system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0, 'fitting_basis':basis}
			)
			assert expected_labels == my_driver.system.labels

def test_with_hip_iad():
	'''
	Make sure the labels are generated correctly for all of the basis sets for the configuration of
	the driver where Hipparcos data is supplied. Again, for the XYZ basis, expect there to be errors thrown.
	'''
	filename = "{}/betaPic.csv".format(DATADIR)
	hip_num = '027321'
	hipp_filename = "{}/HIP{}.d".format(DATADIR, hip_num)

	num_secondary_bodies = 1
	system_mass = 1.75
	mass_err = 0
	plx = 51.44
	plx_err = 0.12

	myHip = hipparcos.HipparcosLogProb(hipp_filename, hip_num, num_secondary_bodies)

	hip_labels = ['pm_ra', 'pm_dec', 'alpha0', 'delta0']

	for basis in basis_names:
		if basis != 'SemiAmp':
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx'] + hip_labels + ['m1', 'm0']
		else:
			expected_labels = [item + '1' for item in basis_names[basis]] + ['plx'] + hip_labels + ['m0']

		if basis == 'XYZ': # Should throw an error for XYZ basis
			with pytest.raises(Exception) as excinfo:
				my_driver = driver.Driver(
					filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
					system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0, 
					'hipparcos_IAD':myHip, 'fitting_basis':basis}
				)
			assert str(excinfo.value) == "For now, the XYZ basis requires data in RA and DEC offsets."
		else:
			my_driver = driver.Driver(
				filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
				system_kwargs = {'fit_secondary_mass':True, 'tau_ref_epoch':0, 
				'hipparcos_IAD':myHip, 'fitting_basis':basis}
			)
			assert expected_labels == my_driver.system.labels

def test_XYZ():
	'''
	Test the XYZ basis on data that does not throw exceptions using simulated data for
	(1) where single companion is supplied and its mass is not being fitted, (2) a single
	companion is supplied and its mass is being fitted, and (3) two companions are supplied
	and their masses are being fitted.
	'''
	filename = '{}/xyz_test_data.csv'.format(DATADIR)
	data = read_input.read_file(filename)

	# (1) Single Companion (mtot)
	single_data = data[np.where(data['object'] == 1)[0]]
	copy = single_data.copy()
	my_system = system.System(1, single_data, 1.22, 56.89, mass_err=0.05, plx_err=0.12, fitting_basis='XYZ')
	expected_labels = [item + '1' for item in basis_names['XYZ']] + ['plx', 'mtot']
	assert expected_labels == my_system.labels

	# (2) Single Companion (fit companion mass)
	my_system = system.System(1, copy, 1.22, 56.89, mass_err=0.05, plx_err=0.12, fit_secondary_mass=True,
		fitting_basis='XYZ')
	expected_labels = [item + '1' for item in basis_names['XYZ']] + ['plx', 'm1', 'm0']
	assert expected_labels == my_system.labels

	# (3) Two Companions (fit companion masses)
	my_system = system.System(2, data, 1.22, 56.89, mass_err=0.05, plx_err=0.12, fit_secondary_mass=True,
		fitting_basis='XYZ')

	expected_labels = [item + '1' for item in basis_names['XYZ']] + [item + '2' for item in basis_names['XYZ']] + ['plx', 'm1', 'm2', 'm0']
	assert expected_labels == my_system.labels

if __name__ == '__main__':
	test_no_extra_data()
	test_with_rv()
	test_with_hip_iad()
	test_XYZ()