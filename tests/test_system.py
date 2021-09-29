"""
Tests functionality of methods in system.py
"""
import numpy as np
import pytest
import os
import orbitize
import orbitize.read_input as read_input
import orbitize.system as system
import orbitize.results as results


def test_convert_data_table_radec2seppa():
    num_secondary_bodies=1
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
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
    input_file = os.path.join(orbitize.DATADIR, 'test_val_multi.csv')
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

    params = np.array([
        7.2774010e+01, 4.1116819e-02, 5.6322372e-01, 3.5251172e+00, 
        4.2904768e+00, 9.4234377e-02, 4.5418411e+01, 1.4317369e-03, 5.6322372e-01, 
        3.1016846e+00, 4.2904768e+00, 3.4033456e-01, 2.4589758e+01, 1.4849439e+00
    ])

    result = test_system.compute_model(params)

    print(result)

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
    test_convert_data_table_radec2seppa()
    test_radec2seppa()
    test_multi_planets()
    
