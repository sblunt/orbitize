"""
Test the functionality of the RV+astrometry implementation
"""

import numpy as np
import orbitize.lnlike as lnlike
import orbitize.system as system
import orbitize.sampler as sampler
import orbitize.read_input as read_input
import os

def test_rvs():

    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table = read_input.read_file(input_file)
    data_table['object'] = 1
    testSystem = system.System(
        2, data_table, 10., 10.
    )
    params_arr = np.array([[1.,0.5],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[245000., 245000.], [10, 10], [10, 10]])
    model_pred = testSystem.compute_model(params_arr)

if __name__=='__main__':
	test_rvs()