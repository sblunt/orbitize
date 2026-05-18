from orbitize import results
import matplotlib.pyplot as plt
hdf5_filename = "my_posterior_brightness.hdf5"


loaded_results = results.Results()  # Create blank results object for loading
loaded_results.load_results(hdf5_filename)

print(loaded_results.post)

orbit_plot_fig = loaded_results.plot_orbits(
    object_to_plot=1,
    num_orbits_to_plot=100,
    start_mjd=0,   
    sep_pa_end_year=1861  
)
# Save or show the figure
orbit_plot_fig.savefig("reloaded_short_bright_orbit_plot.png")
plt.show()