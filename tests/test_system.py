"""
Tests functionality of methods in system.py
"""
import numpy as np
import pytest
import os

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

def test_radec2seppa():

    ras = np.array([-1, -1, 1, 1])
    decs = np.array([-1, 1, -1, 1])

    pas_expected = np.array([225., 315., 135., 45.])
    pas_expected_180mod = np.array([225., 315., 495., 405.])
    seps_expected = np.ones(4)*np.sqrt(2)

    sep_nomod, pa_nomod = system.radec2seppa(ras, decs)
    sep_180mod, pa_180mod = system.radec2seppa(ras, decs, mod180=True)

    assert sep_nomod ==  pytest.approx(seps_expected, abs=1e-3)
    assert sep_180mod ==  pytest.approx(seps_expected, abs=1e-3)
    assert pa_nomod ==  pytest.approx(pas_expected, abs=1e-3)
    assert pa_180mod ==  pytest.approx(pas_expected_180mod, abs=1e-3)


if __name__ == '__main__':
    test_add_and_clear_results()
    test_convert_data_table_radec2seppa()
    test_radec2seppa()
