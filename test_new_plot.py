import orbitize
import sys
sys.path.append('/home/manduhmia/anaconda3/envs/orbitize_develop/lib/python3.10/site-packages')
import matplotlib as mpl

colors = [mpl.cm.Purples_r, mpl.cm.Blues_r, mpl.cm.Greens_r, mpl.cm.Oranges_r]

fpath = '/home/manduhmia/amanda_ceri/orbit_fitting/orbitize_runs/final_four_planet/corrected_prior_implementation/full_run/final_four_results.hdf5'
fres = orbitize.results.Results()
fres.load_results(fpath)

print(fres.post.shape)

bf, sp = orbitize.plot.plot_n_orbits_new(fres, num_objects=4, cmap_list=colors, nbody_solver=True)