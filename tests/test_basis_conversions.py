import orbitize
import numpy as np
from orbitize import system, read_input, DATADIR

def test_period_basis():
	"""
	For both MCMC and OFTI formats, make the conversion to standard basis and go 
	back to original basis and check to see original params are retrieved. Do 
	this with system mass parameter, single companion, and two companions.
	"""
	# 1. With System Total Mass
	filename = "{}/GJ504.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(
		1, data_table, 1.75, 51.44, mass_err=0.05, plx_err=0.12, 
		fitting_basis='Period'
	)

	num_samples = 100
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)
	sample_copy = samples.copy()

	# MCMC Format
	test = samples[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_period_basis(conversion)
	assert np.allclose(original, sample_copy[:, 0])

	# OFTI Format
	conversions = my_system.basis.to_standard_basis(samples)
	original = my_system.basis.to_period_basis(conversions)
	assert np.allclose(original, sample_copy)

	# 2. Single Body (with RV)
	filename = "{}/HD4747.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(
		1, data_table, 0.84, 53.18, mass_err=0.04, plx_err=0.12, 
		fit_secondary_mass=True, fitting_basis='Period'
	)

	num_samples = 100
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)

	sample_copy = samples.copy()

	# MCMC Format
	test = samples[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_period_basis(conversion)
	assert np.allclose(original, sample_copy[:, 0])

	# 3. Multi Body
	filename = "{}/test_val_multi.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(
		2, data_table, 1.52, 24.76, mass_err=0.15, plx_err=0.64, 
		fit_secondary_mass=True,
		fitting_basis='Period'
	)

	num_samples = 100
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)

	sample_copy = samples.copy()

	# MCMC Format
	test = samples[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_period_basis(conversion)
	assert np.allclose(original, sample_copy[:, 0])

	# OFTI Format
	conversions = my_system.basis.to_standard_basis(samples)
	original = my_system.basis.to_period_basis(conversions)
	assert np.allclose(original, sample_copy)

def test_semi_amp_basis():
	"""
	For both MCMC and OFTI param formats, make the conversion to the standard 
	basis from semi-amplitude and back to verify the valdity of conversions. Do 
	this with a single companion and with two companions.
	"""
	# 1. Single Body (with RV)
	filename = "{}/HD4747.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(
		1, data_table, 0.84, 53.18, mass_err=0.04, plx_err=0.12, 
		fit_secondary_mass=True, fitting_basis='SemiAmp'
	)

	num_samples = 100
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)

	sample_copy = samples.copy()

	# MCMC Format
	test = samples[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_semi_amp_basis(conversion)

	assert np.allclose(original, sample_copy[:, 0])

	# 2. Multi Body
	filename = "{}/test_val_multi.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(
		2, data_table, 1.52, 24.76, mass_err=0.15, plx_err=0.64, 
		fit_secondary_mass=True, fitting_basis='SemiAmp'
	)

	num_samples = 100
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)

	sample_copy = samples.copy()

	# MCMC Format
	test = samples[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_semi_amp_basis(conversion)
	assert np.allclose(original, sample_copy[:, 0])

def test_xyz_basis():
	"""
	For both MCMC and OFTI param formats, make the conversion to the standard 
	basis from XYZ basis and back to verify the valdity of conversions. Do this 
	with a single companion and with two companions.
	"""
	# 1. Single Body
	filename = '{}/xyz_test_data.csv'.format(DATADIR)
	data = read_input.read_file(filename)
	single = data[np.where(data['object'] == 1)[0]]
	my_system = system.System(
		1, single, 1.22, 56.89, mass_err=0.05, plx_err=0.12, fitting_basis='XYZ'
	)

	num_samples = 1000 # Do more samples to be safe
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)

	sample_copy = samples.copy()
	conversion = my_system.basis.to_standard_basis(samples)
	locs = np.where((conversion[1, :] >= 1.0) | (conversion[1, :] < 0.))[0]
	sample_copy = np.delete(sample_copy, locs, axis=1)

	# Test MCMC
	test = sample_copy[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_xyz_basis(conversion)
	assert np.allclose(original, sample_copy[:, 0])

	# Test OFTI
	conversions = my_system.basis.to_standard_basis(sample_copy.copy())
	original = my_system.basis.to_xyz_basis(conversions)
	assert np.allclose(original, sample_copy)

	# 2. Multi Body
	my_system = system.System(
		2, data, 1.22, 56.89, mass_err=0.05, plx_err=0.12, fitting_basis='XYZ'
	)

	num_samples = 1000 # Do more samples to be safe
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)

	sample_copy = samples.copy()
	conversion = my_system.basis.to_standard_basis(samples)
	locs = np.where(
		(conversion[[1, 7], :] >= 1.0) | (conversion[[1, 7], :] < 0.)
	)[1]
	locs = np.unique(locs)
	sample_copy = np.delete(sample_copy, locs, axis=1)

	# Test MCMC
	test = sample_copy[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_xyz_basis(conversion)
	assert np.allclose(original, sample_copy[:, 0])

	# Test OFTI	
	conversions = my_system.basis.to_standard_basis(sample_copy.copy())
	original = my_system.basis.to_xyz_basis(conversions)
	assert np.allclose(original, sample_copy)

if __name__ == '__main__':
	test_period_basis()
	test_semi_amp_basis()
	test_xyz_basis()