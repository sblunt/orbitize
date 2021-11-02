from orbitize import read_input, system, priors, sampler,results,kepler

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import os

"""
Simulates RV data from multiple instruments and relative astroemtric data 
from a single instrument, then runs an orbit-fit and recovers the input
parameters.

Written: Vighnesh Nagpal, 2021
"""

def plot_rv(epochs,rvs):
    plt.plot(epochs,rvs)
    plt.savefig('rv_trend')
    plt.close()

def plot_astro(ras,decs):
    plt.plot(ras,decs)
    plt.axis("equal")
    plt.savefig('orbit_trend')
    plt.close()

def gen_data():
    '''
    Simulates radial velocity and astrometric data for a test system.

    Returns: 
        (rvs,rv_epochs): Tuple of generated radial velocity measurements (rvs) and their corresponding 
                         measurement epochs (rv_epochs)

        (ras,decs,astro_epochs): Tuple containing simulated astrometric measurements (ras, decs) 
                                 and the corresponding measurement epochs (astro_epochs)


    '''
    #set parameters for the synthetic data
    sma=1
    inc=np.pi/2
    ecc=0.2
    aop=np.pi/4
    pan=np.pi/4
    tau=0.4
    plx=50
    mass_for_kamp=0.1
    mtot=1.1
    #epochs and errors for rv 
    rv_epochs=np.linspace(51544,52426,200)
    #epochs and errors for astrometry
    astro_epochs=np.linspace(51500,52500,10)
    astro_err=0
    #generate rv trend
    rvset=kepler.calc_orbit(rv_epochs,sma,ecc,inc,aop,pan,tau,plx,mtot,mass_for_Kamp=mass_for_kamp)
    rvs=rvset[2]
    #generate predictions for astrometric epochs
    astro_set=kepler.calc_orbit(astro_epochs,sma,ecc,inc,aop,pan,tau,plx,mtot,mass_for_Kamp=mass_for_kamp)
    ras,decs=astro_set[0],astro_set[1]
    #return model generations
    return (rvs,rv_epochs),(ras,decs,astro_epochs)


def scat_model(rvs,calibration_terms):
    '''
    Function that adds scatter to RV data based on provided calibration terms (gamma, sigma)
    that are unique for each instrument in the dataset.

    Args: 
        rvs (array): Array of radial velocity measurements
        calibration_terms (tuple): Tuple of the form: (gamma_instrument1,jit_instrument1,
                                                       gamma_instrument2,jit_instrument2, 
                                                       rv_err)

    returns:
        scat_rvs (array): Array of RV measurements with scatter added 
        errors (array): Array of measurement uncertainties the RV measurements

    '''
    gam1,jit1,gam2,jit2,rv_err=calibration_terms
    #create empty arrays to be filled with data from each inst +respective jit and sigmas
    length=int(len(rvs)/2)
    off_1=np.zeros(length)
    off_2=np.zeros(length)
    #create an array of normally sampled jitters for each instruments

    errors1=np.abs(rv_err*np.random.randn(length))
    errors2=np.abs(rv_err*np.random.randn(length))


    jscat1=np.random.randn(length)*np.sqrt(jit1**2+errors1**2)
    jscat2=np.random.randn(length)*np.sqrt(jit2**2+errors2**2)
    #fill off_1 and off_2
    off_1[:]=rvs[:length]
    off_2[:]=rvs[length:]
    #add scatters and gammas for first instrument
    off_1+=gam1
    off_1+=jscat1
    #add scatters and gammas for second instrument
    off_2+=gam2
    off_2+=jscat2
    #put em together
    scat_rvs=np.concatenate([off_1,off_2])
    #put measurement uncertainties together
    errors=np.concatenate([errors1,errors2])
    return scat_rvs,errors


def make_csv(fname,rv_epochs,model,astr_info,errors):
    '''
    Takes the data generated and saves it as an orbitize-compatible csv file.

    '''
    #unpack astrometric info
    ras,decs,astro_epochs=astr_info
    #actually make csv
    frame=[]
    for i,val in enumerate(rv_epochs):
        if i<100:
            obs=[val,0,model[i],errors[i],None,None,None,None,'tel_1']
        else:
            obs=[val,0,model[i],errors[i],None,None,None,None,'tel_2']
        frame.append(obs)
    for i,val in enumerate(astro_epochs):
        obs=[val,1,None,None,ras[i],0,decs[i],0,'default']
        frame.append(obs)
    df=pd.DataFrame(frame, columns = ['epoch', 'object','rv','rv_err','raoff','raoff_err','decoff','decoff_err','instrument'])
    df.set_index('epoch', inplace=True)
    df.to_csv(fname)

def run_fit(fname):
    '''
    Runs the orbit fit! Saves the resultant posterior, orbit plot and corner plot

    args: 
        fname (str): Path to the data file. 

    '''
    
    #parameters for the system 
    num_planets=1
    data_table = read_input.read_file(fname)
    m0 = 1.0
    mass_err = 0.01
    plx=50
    plx_err=0.01
    #initialise a system object
    sys = system.System(
        num_planets, data_table, m0,
        plx, mass_err=mass_err, plx_err=plx_err,fit_secondary_mass=True
    )
    #MCMC parameters
    n_temps=5
    n_walkers=1000
    n_threads=5
    total_orbits_MCMC=5000 # n_walkers x num_steps_per_walker
    burn_steps=1
    thin=1
    #set up sampler object and run it 
    mcmc_sampler = sampler.MCMC(sys,n_temps,n_walkers,n_threads)
    orbits = mcmc_sampler.run_sampler(total_orbits_MCMC, burn_steps=burn_steps, thin=thin)
    myResults=mcmc_sampler.results
    try:
        save_path = '.'
        filename  = 'post.hdf5'
        hdf5_filename=os.path.join(save_path,filename)
        myResults.save_results(hdf5_filename)  # saves results object as an hdf5 file
    except:
        print("Something went wrong while saving the results")
    finally:      
        corner_figure=myResults.plot_corner()
        corner_name='corner.png'
        corner_figure.savefig(corner_name)

        orbit_figure=myResults.plot_orbits(rv_time_series=True)
        orbit_name='joint_orbit.png'
        orbit_figure.savefig(orbit_name)  

    print("Done!")

if __name__=='__main__':
    
    rv_info,astr_info=gen_data()
    rvs,rv_epochs=rv_info

    # set gammas and jitters

    calibration_terms=(0.7,0.009,-0.3,0.006,0.002)

    #add scatter to model
    model,errors=scat_model(rvs,calibration_terms)
    
    #save this to a new file
    fname='./simulated_data.csv'
    make_csv(fname,rv_epochs,model,astr_info,errors)

    # run orbit fit
    run_fit(fname)

    # delete CSV
    os.remove("demofile.txt")



    