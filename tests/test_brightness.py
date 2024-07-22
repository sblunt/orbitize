"""
Tests for the reflected-light calculation in system.compute_all_orbits
"""

from orbitize import system
from orbitize import DATADIR, read_input
import os
import numpy as np


def test_brightness_calculation():

    num_secondary_bodies = 1
    input_file = os.path.join(DATADIR, "GJ504.csv")
    data_table = read_input.read_file(input_file)
    system_mass = 1.47
    plx = 24.30

    test_system = system.System(num_secondary_bodies, data_table, system_mass, plx)

    params = np.array(
        [
            7.2774010e01,
            4.1116819e-02,
            5.6322372e-01,
            3.5251172e00,
            4.2904768e00,
            9.4234377e-02,
            4.5418411e01,
            1.4317369e-03,
            5.6322372e-01,
            3.1016846e00,
            4.2904768e00,
            3.4033456e-01,
            2.4589758e01,
            1.4849439e00,
        ]
    )

    ra, dec, vz, brightness = test_system.compute_all_orbits(params)
