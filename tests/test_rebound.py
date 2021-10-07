from matplotlib import pyplot as plt
from astropy.time import Time
from orbitize import DATADIR, kepler
from orbitize.system import System
from orbitize.read_input import read_file
import astropy.units as u
import numpy as np
import orbitize.nbody as nbody

#Test Data
mass_in_mjup = 10
mB_Jup = 7
mass_in_msun = mass_in_mjup * u.Mjup.to(u.Msun)
massB = mB_Jup * u.Mjup.to(u.Msun)
m_pl = np.array([mass_in_msun, mass_in_msun, mass_in_msun, massB])

#From System HR-8799
#NOTE planets get closer to the star in alphabetical order, i.e. B is farthest, E is closest
sma = np.array([16, 26,43, 71]) 
ecc = np.array([.12, .13, .02, .02])
inc = np.array([0.47123, 0.47123, 0.47123, 0.47123])
aop = np.array([1.91986, 0.29670, 1.16937, 1.5184])
pan = np.array([1.18682, 1.18682, 1.18682, 1.18682])
tau = np.array([0.71, 0.79, 0.50, 0.54])
plx = np.array([7])
mtot = np.array([1.49])
tau_ref_epoch = 0
years = 365.25*5

# need a properly formatted data table for the code to run but it doesn't 
# matter what's in it
input_file = '{}/GJ504.csv'.format(DATADIR)
data_table = read_file(input_file)
num_secondary_bodies = 4

epochs = Time(np.linspace(2020, 2022, num=int(1000)), format='decimalyear').mjd

sma1 = sma[0]
ecc1 = ecc[0]
inc1 = inc[0]
aop1 = aop[0]
pan1 = pan[0]
tau1 = tau[0]
sma2 = sma[1]
ecc2 = ecc[1]
inc2 = inc[1]
aop2 = aop[1]
pan2 = pan[1]
tau2 = tau[1]
sma3 = sma[2]
ecc3 = ecc[2]
inc3 = inc[2]
aop3 = aop[2]
pan3 = pan[2]
tau3 = tau[2]
sma4 = sma[3]
ecc4 = ecc[3]
inc4 = inc[3]
aop4 = aop[3]
pan4 = pan[3]
tau4 = tau[3]
m1 = m_pl[0]
m2 = m_pl[1]
m3 = m_pl[2]
m4 = m_pl[3]
m_st = mtot-sum(m_pl)

hr8799_sys = System(
    num_secondary_bodies, data_table, m_st,
    plx, fit_secondary_mass=True, tau_ref_epoch=tau_ref_epoch
)

params_arr = np.array([
    sma1, ecc1, inc1, aop1, pan1, tau1,
    sma2, ecc2, inc2, aop2, pan2, tau2,
    sma3, ecc3, inc3, aop3, pan3, tau3,
    sma4, ecc4, inc4, aop4, pan4, tau4,
    plx,
    m1, m2, m3, m4, m_st
])

def test_8799_rebound_vs_kepler(plotname=None):
    """
    Test that for the HR 8799 system between 2020 and 2022, the rebound and
    keplerian (with epicycle approximation) backends produce similar results.
    """

    assert hr8799_sys.track_planet_perturbs
    
    rra, rde, _ = hr8799_sys.compute_all_orbits(params_arr, epochs=epochs, comp_rebound=True)
    kra, kde, _ = hr8799_sys.compute_all_orbits(params_arr, epochs=epochs, comp_rebound=False)

    delta_ra = abs(rra-kra[:,:,0])
    delta_de = abs(rde-kde[:,:,0])
    yepochs = Time(epochs, format='mjd').decimalyear

    # check that the difference between these two solvers is smaller than 
    #   GRAVITY precision over the next two years
    gravity_precision_mas = 100 * 1e-3
    assert np.all(delta_ra < gravity_precision_mas)
    assert np.all(delta_de < gravity_precision_mas)

    # check that the difference between these two solvers is larger than 
    #   10 uas, though
    assert np.max(delta_ra) > 10 * 1e-3
    assert np.max(delta_de) > 10 * 1e-3

    if plotname is not None:

        # plot 1: RA & Dec differences over time
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Massive Orbits in Rebound vs. Orbitize approx.')

        ax1.plot(yepochs, delta_ra[:,0], 'black', label = 'Star')
        ax2.plot(yepochs, delta_de[:,0], 'dimgray', label = 'Star')

        ax1.plot(yepochs, delta_ra[:,1], 'brown', label = 'Planet E: RA offsets') #first planet
        ax2.plot(yepochs, delta_de[:,1], 'red', label = 'Planet E: Dec offsets')

        ax1.plot(yepochs, delta_ra[:,2], 'coral', label = 'Planet D: RA offsets') #second planet
        ax2.plot(yepochs, delta_de[:,2], 'orange', label = 'Planet D: Dec offsets')

        ax1.plot(yepochs, delta_ra[:,3], 'greenyellow', label = 'Planet C: RA offsets') #third planet
        ax2.plot(yepochs, delta_de[:,3], 'green', label = 'Planet C: Dec offsets')

        ax1.plot(yepochs, delta_ra[:,4], 'dodgerblue', label = 'Planet B: RA offsets') #fourth planet
        ax2.plot(yepochs, delta_de[:,4], 'blue', label = 'Planet B: Dec offsets')
        
        plt.xlabel('year')
        plt.ylabel('milliarcseconds')
        ax1.legend()
        ax2.legend()
        plt.savefig('{}_differences.png'.format(plotname), dpi=250)

        # plot 2: orbit tracks over time
        plt.figure()
        plt.plot(kra[:,1:5,0], kde[:,1:5,0], 'indigo', label = 'Orbitize approx.')
        plt.plot(kra[-1,1:5,0], kde[-1,1:5,0],'o')
        
        plt.plot(rra, rde, 'r', label = 'Rebound', alpha = 0.25)
        plt.plot(rra[-1], rde[-1], 'o', alpha = 0.25)
            
        plt.plot(0, 0, '*')
        plt.legend()
        plt.savefig('{}_orbittracks.png'.format(plotname), dpi=250)

        # plot 3: primary orbit track over time
        plt.figure()
        plt.plot(kra[:,0,0], kde[:,0,0], 'indigo', label = 'Orbitize approx.')
        plt.plot(kra[-1,0,0], kde[-1,0,0],'o')
        
        plt.plot(rra[:,0], rde[:,0], 'r', label = 'Rebound', alpha = 0.25)
        plt.plot(rra[-1,0], rde[-1,0], 'o', alpha = 0.25)
            
        plt.plot(0, 0, '*')
        plt.legend()
        plt.savefig('{}_primaryorbittrack.png'.format(plotname), dpi=250)

if __name__=='__main__':
    test_8799_rebound_vs_kepler(plotname='hr8799_diffs')
