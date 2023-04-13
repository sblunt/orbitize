from orbitize import system, priors, sampler
import numpy as np
import time
from orbitize.system import generate_synthetic_data
import time

def func(orbit_frac):
    """
    Args:
        orbit_frac (float): percentage of orbit covered by the synthetic data
    Returns:
        3-tuple:
            -array: posterior samples
            -float: run time
            -int: number of iterations it took to converge on posterior
    """
    

    # generate data
    mtot = 1.2 # total system mass [M_sol]
    plx = 60.0 # parallax [mas
    # sma = 2.3
    data_table = generate_synthetic_data(orbit_frac, mtot, plx, num_obs=30)

    # initialize orbitize System object
    sys = system.System(1, data_table, mtot, plx)
    print(data_table)
    lab = sys.param_idx

    # set all parameters except eccentricity to fixed values for testing
    sys.sys_priors[lab['inc1']] = np.pi/4
    # sys.sys_priors[lab['sma1']] = sma
    # sys.sys_priors[lab['aop1']] = np.pi/4 
    # sys.sys_priors[lab['pan1']] = np.pi/4
    # sys.sys_priors[lab['tau1']] = 0.8  
    # sys.sys_priors[lab['plx']] = plx
    # sys.sys_priors[lab['mtot']] = mtot

    #record start time
    start = time.time()
    
    # run nested sampler
    nested_sampler = sampler.NestedSampler(sys)
    samples, num_iter = nested_sampler.run_sampler(static = True, 
    bound = 'multi')

    #calculate script run time
    execution_time = (time.time() - start) / 60 #minutes

    return (samples, execution_time, num_iter)