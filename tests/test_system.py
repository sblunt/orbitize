"""
Tests functionality of methods in system.py
"""
import os
import numpy as np

import orbitize.read_input as read_input
import orbitize.system as system
import orbitize.results as results

def test_add_and_clear_results():
    num_secondary_bodies=1
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table=read_input.read_file(input_file)
    system_mass=1.0
    plx=10.0
    mass_err=0.1
    plx_err=1.0
    # Initialize System object
    test_system = system.System(
        num_secondary_bodies, data_table, system_mass,
        plx, mass_err=mass_err, plx_err=plx_err
    )
    # Initialize dummy results.Results object
    test_results = results.Results()
    # Add one result object
    test_system.add_results(test_results)
    assert len(test_system.results)==1
    # Adds second result object
    test_system.add_results(test_results)
    assert len(test_system.results)==2
    # Clears result objects
    test_system.clear_results()
    assert len(test_system.results)==0
    # Add one more result object
    test_system.add_results(test_results)
    assert len(test_system.results)==1


def test_convert_data_table_radec2seppa():
    num_secondary_bodies=1
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table=read_input.read_file(input_file)
    system_mass=1.0
    plx=10.0
    mass_err=0.1
    plx_err=1.0
    # Initialize System object
    test_system = system.System(
        num_secondary_bodies, data_table, system_mass,
        plx, mass_err=mass_err, plx_err=plx_err
    )
    test_system.convert_data_table_radec2seppa()

def test_multi_planets():
    """
    Test using some data for HR 8799 b and c
    """
    num_secondary_bodies=2
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val_multi.csv')
    data_table=read_input.read_file(input_file)
    system_mass=1.47
    plx=24.30
    mass_err=0.11
    plx_err=0.7
    # Initialize System object, use mjd=50000 to be consistent with Wang+2018
    test_system = system.System(
        num_secondary_bodies, data_table, system_mass,
        plx, mass_err=mass_err, plx_err=plx_err, tau_ref_epoch=50000
    )

    params = np.array([7.2774010e+01, 4.1116819e-02, 5.6322372e-01, 3.5251172e+00, 4.2904768e+00, 9.4234377e-02, 
              4.5418411e+01, 1.4317369e-03, 5.6322372e-01, 3.1016846e+00, 4.2904768e+00, 3.4033456e-01,
              2.4589758e+01, 1.4849439e+00])

    result = test_system.compute_model(params)

    print(result)

if __name__ == '__main__':
    #test_add_and_clear_results()
    #test_convert_data_table_radec2seppa()
    test_multi_planets()
