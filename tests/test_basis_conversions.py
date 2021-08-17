import orbitize
import numpy as np
from orbitize import system, read_input, DATADIR

def test_period_basis():
	'''
	For both MCMC and OFTI formats, make the conversion to standard basis and go back to original
	basis and check to see original params are retrieved.
	'''

	# 1. With System Total Mass
	filename = "{}/GJ504.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(1, data_table, 1.75, 51.44, mass_err=0.05, plx_err=0.12, fitting_basis='Period')

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
	my_system = system.System(1, data_table, 0.84, 53.18, mass_err=0.04, plx_err=0.12, fit_secondary_mass='True', 
		fitting_basis='Period')

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

	# 3. Multi Body
	filename = "{}/test_val_multi.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(2, data_table, 1.52, 24.76, mass_err=0.15, plx_err=0.64, fit_secondary_mass='True',
		fitting_basis='Period')

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
	# 1. Single Body (with RV)
	filename = "{}/HD4747.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(1, data_table, 0.84, 53.18, mass_err=0.04, plx_err=0.12, fit_secondary_mass='True', 
		fitting_basis='SemiAmp')

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

	# OFTI Format
	conversion = my_system.basis.to_standard_basis(samples)
	original = my_system.basis.to_semi_amp_basis(conversion)
	assert np.allclose(original, sample_copy)

	# 2. Multi Body
	filename = "{}/test_val_multi.csv".format(DATADIR)
	data_table = read_input.read_file(filename)
	my_system = system.System(2, data_table, 1.52, 24.76, mass_err=0.15, plx_err=0.64, fit_secondary_mass='True',
		fitting_basis='SemiAmp')

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

	# OFTI Format
	conversion = my_system.basis.to_standard_basis(samples)
	original = my_system.basis.to_semi_amp_basis(conversion)
	assert np.allclose(original, sample_copy)

def test_xyz_basis():
	# 1. Single Body
	filename = '{}/xyz_test_data.csv'.format(DATADIR)
	data = read_input.read_file(filename)
	single = data[np.where(data['object'] == 1)[0]]
	my_system = system.System(1, single, 1.75, 51.44, mass_err=0.05, plx_err=0.12, fitting_basis='XYZ')

	num_samples = 100
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)

	sample_copy = samples.copy()

	# OFTI Format
	conversions = my_system.basis.to_standard_basis(samples)

	# Filter out all orbits with 'nan' ecc or ecc >= 1
	locs = np.where(np.logical_or(np.isnan(conversions[1, :]), conversions[1, :] >= 1))[0]
	conversions = np.delete(conversions, locs, axis=1)
	sample_copy = np.delete(sample_copy, locs, axis=1)

	original = my_system.basis.to_xyz_basis(conversions)
	assert np.allclose(original, sample_copy)

	# MCMC Format
	test = sample_copy[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_xyz_basis(conversion)
	assert np.allclose(original, sample_copy[:, 0])

	# 2. Multi-Body
	my_system = system.System(2, data, 1.75, 51.44, mass_err=0.05, plx_err=0.12, fitting_basis='XYZ')
	num_samples = 100
	samples = np.empty([len(my_system.sys_priors), num_samples])
	for i in range(len(my_system.sys_priors)):
		if hasattr(my_system.sys_priors[i], "draw_samples"):
			samples[i, :] = my_system.sys_priors[i].draw_samples(num_samples)
		else:
			samples[i, :] = my_system.sys_priors[i] * np.ones(num_samples)

	sample_copy = samples.copy()

	# OFTI Format
	indices_to_remove = np.empty(0)
	conversions = my_system.basis.to_standard_basis(samples)
	for i in range(2):
		index = (i * 6) + 1
		to_remove = np.where(np.logical_or(np.isnan(conversions[index, :]), conversions[index, :] >= 1))[0]
		indices_to_remove = np.union1d(indices_to_remove, to_remove)

	indices_to_remove = indices_to_remove.astype(int)
	conversions = np.delete(conversions, indices_to_remove, axis=1)
	sample_copy = np.delete(sample_copy, indices_to_remove, axis=1)

	original = my_system.basis.to_xyz_basis(conversions)
	assert np.allclose(original, sample_copy)

	# MCMC Format
	test = sample_copy[:, 0].copy()
	conversion = my_system.basis.to_standard_basis(test)
	original = my_system.basis.to_xyz_basis(conversion)
	assert np.allclose(original, sample_copy[:, 0])


if __name__ == '__main__':
	test_period_basis()
	test_semi_amp_basis()