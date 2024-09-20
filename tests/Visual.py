from orbitize import system, read_input, DATADIR
import os
import numpy as np
import matplotlib.pyplot as plt


num_secondary_bodies = 1

input_file = os.path.join(DATADIR, "GJ504.csv")
data_table = read_input.read_file(input_file)

system_mass = 1.47
plx = 24.30

test_system = system.System(num_secondary_bodies, data_table, system_mass, plx)

params_arr = np.array(
    [
        10.0,  # sma
        0.9,  # ecc
        np.radians(30),  # inc
        np.radians(60),  # aop
        np.radians(120),  # pan
        0.0,  # tau
        plx,
        system_mass,
    ]
)
epochs = np.linspace(0, 365 * 30, int(1e3))
ra, dec, vz, brightness = test_system.compute_all_orbits(params_arr, epochs=epochs)

fig, ax = plt.subplots(2, 1, figsize=(5, 10))

ax[0].scatter(
    epochs,
    brightness,
    color=plt.cm.RdYlBu((epochs - epochs[0]) / (epochs[-1] - epochs[0])),
)

ax[1].scatter(
    ra[:, 1, :],
    dec[:, 1, :],
    color=plt.cm.RdYlBu((epochs - epochs[0]) / (epochs[-1] - epochs[0])),
)

ax[1].scatter([0], [0], color="red")

ax[1].axis("equal")

plt.savefig("visual4farrah.png")