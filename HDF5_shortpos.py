from orbitize import results
import matplotlib.pyplot as plt
hdf5_filename = "/home/fmolina/orbitize/tests/my_posterior_shortpos.hdf5"


loaded_results = results.Results()  # Create blank results object for loading
loaded_results.load_results(hdf5_filename)

print(loaded_results.labels)


orbit_plot_fig = loaded_results.plot_orbits(
    object_to_plot=1,
    num_orbits_to_plot=100,
    start_mjd=0,   
    sep_pa_end_year=1859  
)
# save fig
orbit_plot_fig.savefig("/home/fmolina/orbitize/reloaded_shortpos_plot.png")
plt.show()