from orbitize import driver, DATADIR, results
import corner
import matplotlib.pyplot as plt
import numpy as np

myDriver = driver.Driver(
    '{}/GJ504.csv'.format(DATADIR), # data file
    'OFTI',        # choose from: ['OFTI', 'MCMC']
    1,             # number of planets in system
    1.22,          # total mass [M_sun]
    56.95,         # system parallax [mas]
    mass_err=0.08, # mass error [M_sun]
    plx_err=0.26   # parallax error [mas]
)

if __name__ == '__main__':
    myDriver.sampler.run_sampler(100_000, num_samples=int(1e6), num_cores=1)

    # plot the results
    myResults = myDriver.sampler.results
    myResults.save_results('GJ504.hdf5')

    myResults = results.Results()
    myResults.load_results('GJ504.hdf5')

    unifmanomResults = results.Results()
    unifmanomResults.load_results('GJ504_uniform_meananno.hdf5')

    range = [(0,200), (0,1), (0, np.pi), (0, 2*np.pi), (0, 2*np.pi), (0,1), (56, 58), (.9, 1.5)]

    fig = corner.corner(myResults.post, color='red', bins=100, labels=unifmanomResults.system.labels, range=range)
    corner.corner(unifmanomResults.post, color='blue', fig=fig, bins=100,  range=range)

    plt.savefig('GJ504.png', dpi=250)



