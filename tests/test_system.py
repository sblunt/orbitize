"""
Tests functionality of methods in system.py
"""

import orbitize.read_input as read_input
import orbitize.system as system
import orbitize.results as results
import os

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

if __name__ == '__main__':
    test_add_and_clear_results()
