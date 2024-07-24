"""
Tests for the reflected-light calculation in system.compute_all_orbits
"""

from orbitize import system
from orbitize import DATADIR, read_input
import os
import numpy as np


def test_brightness_calculation():

    num_secondary_bodies = 1

    # input_file = os.path.join(DATADIR, "GJ504.csv")
    input_file = os.path.join(DATADIR, "betaPic.csv")
    data_table = read_input.read_file(input_file)

    times = data_table["epoch"].value

    system_mass = 1.47
    plx = 24.30

    test_system = system.System(num_secondary_bodies, data_table, system_mass, plx)

    print(test_system.param_idx)

    params = np.array(
        [
            10.0,
            0.1,
            np.radians(89),
            np.radians(21),
            np.radians(31),
            0.0,  # note: I didn't convert tau here, just picked random number
            51.5,
            1.75,
        ]
    )

    ra, dec, vz, brightness = test_system.compute_all_orbits(params)

    # TODO (farrah): make plot of brightness vs time


def test_read_input_with_brightness():

    # TODO (farrah): use code above as inspiration to read in a csv file with a brightness column

    print("hello! :D ")


if __name__ == "__main__":
    test_brightness_calculation()
    test_read_input_with_brightness()
