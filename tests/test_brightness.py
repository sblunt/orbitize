"""
Tests for the reflected-light calculation in system.compute_all_orbits
"""

from orbitize import system, sampler
from orbitize import DATADIR, read_input
import os
import numpy as np
import matplotlib.pyplot as plt


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
            0.3,
            np.radians(89),
            np.radians(21),
            np.radians(31),
            0.0,  
            51.5,
            1.75,
        ]
    )

    ra, dec, vz, brightness = test_system.compute_all_orbits(params)

    plt.figure()
    plt.scatter(times, brightness)
    plt.xlabel("Time [dy]", fontsize=18)
    plt.ylabel("Brightness", fontsize=18)
    plt.savefig("Test_brightness.png")


def test_read_input_with_brightness():

    num_secondary_bodies = 1

    input_file = os.path.join(DATADIR, "reflected_light_example.csv")

    data_table = read_input.read_file(input_file)

    times = data_table["epoch"].value
    brightness_values = data_table["brightness"].value

    print(data_table)

    # TODO (Farrah): add a test that asserts the brightness column of the data table is 
    # what you expect (hint: check in the reflected_light_example.csv to see what
    # the brightness values should be
def test_assert_brightness():

    num_secondary_bodies = 1

    input_file = os.path.join(DATADIR, "reflected_light_example.csv")

    data_table = read_input.read_file(input_file)

    brightness_values = data_table["brightness"].value
    assert 'brightness' in data_table.columns, "Brightness column does not exist"
    print("The assert works!")
    print (brightness_values)
    x = "welcome"

    #if condition returns False, AssertionError is raised:
    assert x != "hello", "This is meant to say welcome"


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
    #test_read_input_with_brightness()
    test_assert_brightness()
    # test_compute_posteriors()
