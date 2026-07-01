"""
Tests for the reflected-light calculation in system.compute_all_orbits
"""

from orbitize import system, sampler, plot
from orbitize import DATADIR, read_input
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
import pytest


def test_brightness_calculation():

    num_secondary_bodies = 1

    input_file = os.path.join(DATADIR, "reflected_light_example.csv")
    data_table = read_input.read_file(input_file)

    times = data_table["epoch"].value

    system_mass = 1.47
    plx = 24.30

    test_system = system.System(num_secondary_bodies, data_table, system_mass, plx)

    params = np.array(
        [
            10.0,
            0.3,
            np.radians(89),
            np.radians(21),
            np.radians(70),
            0.0,  
            51.5,
            1.75,
        ]
    )

    ra, dec, vz, brightness = test_system.compute_all_orbits(params)

    expected_brightness = np.zeros((len(times), num_secondary_bodies + 1, 1))
    expected_brightness[:,1,:] = np.array([
        .00040277, .00040277, .00164703, .00164703, .00032383, .00031209, .00029531
    ]).reshape((7,1))

    assert expected_brightness == pytest.approx(brightness, abs=1e-8)


def test_read_input_with_brightness():

    num_secondary_bodies = 1

    input_file = os.path.join(DATADIR, "reflected_light_example.csv")

    data_table = read_input.read_file(input_file)

    assert len(data_table[data_table['quant_type'] == 'brightness']) == 2
    assert len(data_table[data_table['quant_type'] == 'seppa']) == 5

    brightness_data = data_table[data_table['quant_type'] == 'brightness']
    assert np.all(brightness_data['quant1'].value == [0.9, 0.6])
    assert np.all(brightness_data['quant1_err'].value == [0.1, 0.2])
    assert np.all(np.isnan(brightness_data['quant2'].value))
    assert np.all(np.isnan(brightness_data['quant2_err'].value))


def test_compute_posteriors():
    """
    Test that a short mcmc runs to completion 
    """

    num_secondary_bodies = 1

    input_file = os.path.join(DATADIR, "reflected_light_example.csv")
    data_table = read_input.read_file(input_file)

    system_mass = 1.47
    plx = 24.30

    test_system = system.System(num_secondary_bodies, data_table, system_mass, plx)
    test_mcmc = sampler.MCMC(test_system, num_temps=1, num_walkers=30, num_threads=1)
    test_mcmc.run_sampler(1)

    

if __name__ == "__main__":
    test_brightness_calculation()
    # test_read_input_with_brightness()
    # test_compute_posteriors()
