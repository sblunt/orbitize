# Save Chains Corner Plot
import orbitize.results

import matplotlib.pyplot as plt

myResults = orbitize.results.Results()  # create empty Results object
myResults.load_results("my_posterior.hdf5")  # load from file

corner_figure = myResults.plot_corner()