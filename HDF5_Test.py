from orbitize import results
hdf5_filename = "my_posterior.hdf5"


loaded_results = results.Results()  # Create blank results object for loading
loaded_results.load_results(hdf5_filename)

print(loaded_results.post)