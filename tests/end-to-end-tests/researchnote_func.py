from orbitize import system, priors, sampler
import numpy as np
import time
from orbitize.system import generate_synthetic_data

def func(orbit_frac, set_priors = None):
    """
    Args:
        orbit_frac (float): percentage of orbit covered by the synthetic data
        set priors (4-array): inclination (radians), argument of periastron 
        (radians), position angle of nodes (radians), tau
    Returns:
        3-tuple:
            -array: posterior samples
            -float: run time
            -int: number of iterations it took to converge on posterior
    """
    
    # generate data
    plx = 60.0 # [mas]
    mtot = 1.2 # [M_sol]
    data_table, sma = generate_synthetic_data(orbit_frac, mtot, plx, num_obs=30)

    # initialize orbitize System object
    sys = system.System(1, data_table, mtot, plx)
    print(data_table)
    lab = sys.param_idx

    # set specified parameters except eccentricity to fixed values for testing
    if set_priors != None:
        sys.sys_priors[lab['inc1']] = set_priors[0]
        sys.sys_priors[lab['aop1']] = set_priors[1] 
        sys.sys_priors[lab['pan1']] = set_priors[2]
        sys.sys_priors[lab['tau1']] = set_priors[3]  
    else:
        sys.sys_priors[lab['inc1']] = np.pi/4
        sys.sys_priors[lab['aop1']] = np.pi/4 
        sys.sys_priors[lab['pan1']] = np.pi/4
        sys.sys_priors[lab['tau1']] = 0.8

    sys.sys_priors[lab['sma1']] = sma
    sys.sys_priors[lab['plx']] = plx
    sys.sys_priors[lab['mtot']] = mtot
    
    #record start time
    start = time.time()
    
    # run nested sampler
    nested_sampler = sampler.NestedSampler(sys)
    samples, num_iter = nested_sampler.run_sampler(static = True, 
    bound = 'multi')

    #calculate script run time
    execution_time = (time.time() - start) / 60 #minutes

    return (samples, execution_time, num_iter)

if __name__ == '__main__':
    orbit_frac = 95
    samples, exec_time, niter = func(orbit_frac)
    print("execution time (min) is: " + str(exec_time))
    print("iteration number is: " + str(niter))

