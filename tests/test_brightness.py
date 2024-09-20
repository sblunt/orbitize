"""
Tests for the reflected-light calculation in system.compute_all_orbits
"""

from orbitize import system, sampler
from orbitize import DATADIR, read_input
import os
import numpy as np
import matplotlib.pyplot as plt

print('hello')


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

    plt.figure()
    plt.scatter(times, brightness)
    plt.xlabel("Time [dy]")
    plt.ylabel("Brightness")
    plt.savefig("Test_brightness.png")


def test_read_input_with_brightness():

    # TODO (farrah): use code above as inspiration to read in a csv file with a brightness column
    num_secondary_bodies = 1

    # input_file = os.path.join(DATADIR, "GJ504.csv")
    input_file = os.path.join(DATADIR, "betaPic.csv")

    data_table = read_input.read_file(input_file)

    times = data_table["epoch"].value
    brightness_values = data_table["brightness"].value

    # Do we need the rest of this? since the values for time and brightness are given

    print("hello! :D ")


def test_compute_posteriors():

    num_secondary_bodies = 1

    input_file = os.path.join(DATADIR, "GJ504.csv")
    data_table = read_input.read_file(input_file)

    system_mass = 1.47
    plx = 24.30

    test_system = system.System(num_secondary_bodies, data_table, system_mass, plx)

    params_arr = np.array(
        [
            10.0, # sma
            0.3, # ecc
            np.radians(0), # inc
            np.radians(45), # aop
            np.radians(90), # pan
            0.0,  # tau
            51.5, # plx
            1.75, # stellar maxx
        ]
    )
    epochs = np.linspace(0, 365*30, int(1e3))
    ra, dec, vz, brightness = test_system.compute_all_orbits(params_arr, epochs=epochs)

    fig, ax = plt.subplots(2, 1, figsize=(5,10))


    ax[0].scatter(epochs, brightness, color=plt.cm.RdYlBu((epochs-epochs[0])/(epochs[-1] - epochs[0])))
    ax[1].scatter(ra[:,1,:], dec[:,1,:], color=plt.cm.RdYlBu((epochs-epochs[0])/(epochs[-1] - epochs[0])))

    ax[1].axis('equal')
    plt.savefig('visual4farrah.png')

    # model = test_system.compute_model(params_arr)
    # print(model)

    # test_mcmc = sampler.MCMC(test_system, 1, 50, num_threads=1)

    # test_mcmc.run_sampler(10)


if __name__ == "__main__":
    # test_brightness_calculation()
    # test_read_input_with_brightness()
    test_compute_posteriors()

    # Test commit
