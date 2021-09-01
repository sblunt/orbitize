import numpy as np
import os
import emcee

import matplotlib.pyplot as plt
from scipy.stats import norm

from orbitize import DATADIR, read_input, system
from orbitize.hipparcos import HipparcosLogProb

def test_hipparcos_api():
    """
    Check that error is caught for a star with solution type != 5 param, 
    and that doing an RV + Hipparcos IAD fit produces the expected array of 
    Prior objects.
    """

    # check sol type != 5 error message
    hip_num = '25'
    num_secondary_bodies = 1
    iad_file = 'foo' # Doesn't actually matter,
                     # HipparcosLogProb initialization code shouldn't get to here

    try:
        _ = HipparcosLogProb(iad_file, hip_num, num_secondary_bodies)
        assert False, 'Test failed.'
    except ValueError: 
        pass

    # check that RV + Hip gives correct prior array labels
    hip_num = '027321' # beta Pic
    num_secondary_bodies = 1
    iad_file = '{}/HIP{}.d'.format(DATADIR, hip_num)
    myHip = HipparcosLogProb(iad_file, hip_num, num_secondary_bodies)

    input_file = os.path.join(DATADIR, 'HD4747.csv')
    data_table_with_rvs = read_input.read_file(input_file)
    mySys = system.System(
        1, data_table_with_rvs, 1.22, 56.95, mass_err=0.08, plx_err=0.26, 
        hipparcos_IAD=myHip, sampler_str='MCMC', fit_secondary_mass=True
    )

    assert len(mySys.sys_priors) == 15 # 7 orbital params + 2 mass params + 
                                       # 4 Hip nuisance params + 
                                       # 2 RV nuisance params

    assert mySys.labels == [
       'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'plx', 'pm_ra', 'pm_dec', 
       'alpha0', 'delta0', 'gamma_defrv', 'sigma_defrv', 'm1', 'm0'
   ]


def test_iad_refitting():
    """
    Check that refitting the IAD gives posteriors that approximately match
    the official Hipparcos values. Only run the MCMC for a few steps because 
    this is a unit test. 
    """

    post, myHipLogProb = _nielsen_iad_refitting_test(
        iad_loc=DATADIR, burn_steps=10, mcmc_steps=100, saveplot=None
    )

    # check that we get reasonable values for the posteriors of the refit IAD
    # (we're only running the MCMC for a few steps, so these are not strict)
    assert np.isclose(0, np.median(post[:, -1]), atol=0.1)
    assert np.isclose(myHipLogProb.plx0, np.median(post[:, 0]), atol=0.1)

def _nielsen_iad_refitting_test(
    hip_num='027321', saveplot='bPic_IADrefit.png', 
    iad_loc='/data/user/sblunt/HipIAD', burn_steps=100, mcmc_steps=5000
):
    """
    Reproduce the IAD refitting test from Nielsen+ 2020 (end of Section 3.1).
    The default MCMC parameters are what you'd want to run before using 
    the IAD for a new system. This fit uses 100 walkers. 

    Args:
        hip_num (str): Hipparcos ID of star. Available on Simbad.
        saveplot (str): what to save the summary plot as. If None, don't make a 
            plot
        iad_loc (str): path to the directory containing the IAD file.
        burn_steps (int): number of MCMC burn-in steps to run.
        mcmc_steps (int): number of MCMC production steps to run.

    Returns:
        tuple of:
            numpy.array of float: n_steps x 5 array of posterior samples
            orbitize.hipparcos.HipparcosLogProb: the object storing relevant
                metadata for the performed Hipparcos IAD fit
    """
    
    num_secondary_bodies = 0
    iad_file = '{}/HIP{}.d'.format(iad_loc, hip_num)
    myHipLogProb = HipparcosLogProb(
        iad_file, hip_num, num_secondary_bodies, renormalize_errors=True
    )
    n_epochs = len(myHipLogProb.epochs)

    def log_prob(model_pars):
        ra_model = np.zeros(n_epochs)
        dec_model = np.zeros(n_epochs)
        lnlike = myHipLogProb.compute_lnlike(ra_model, dec_model, model_pars)
        return lnlike
    
    ndim, nwalkers = 5, 100

    # initialize walkers
    # (fitting only plx, mu_a, mu_d, alpha_H0, delta_H0)
    p0 = np.random.randn(nwalkers, ndim)

    # plx
    p0[:,0] *= myHipLogProb.plx0_err
    p0[:,0] += myHipLogProb.plx0

    # PM
    p0[:,1] *= myHipLogProb.pm_ra0
    p0[:,1] += myHipLogProb.pm_ra0_err
    p0[:,2] *= myHipLogProb.pm_dec0
    p0[:,2] += myHipLogProb.pm_dec0_err

    # set up an MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    print('Starting burn-in!')
    state = sampler.run_mcmc(p0, burn_steps)
    sampler.reset()
    print('Starting production chain!')
    sampler.run_mcmc(state, mcmc_steps)


    if saveplot is not None:
        _, axes = plt.subplots(5, figsize=(5,12))

        # plx
        xs = np.linspace(
            myHipLogProb.plx0 - 3 * myHipLogProb.plx0_err, 
            myHipLogProb.plx0 + 3 * myHipLogProb.plx0_err,
            1000
        )
        axes[0].hist(sampler.flatchain[:,0], bins=50, density=True, color='r')
        axes[0].plot(
            xs, norm(myHipLogProb.plx0, myHipLogProb.plx0_err).pdf(xs), 
            color='k'
        )
        axes[0].set_xlabel('plx [mas]')

        # PM RA
        xs = np.linspace(
            myHipLogProb.pm_ra0 - 3 * myHipLogProb.pm_ra0_err, 
            myHipLogProb.pm_ra0 + 3 * myHipLogProb.pm_ra0_err,
            1000
        )
        axes[1].hist(sampler.flatchain[:,1], bins=50, density=True, color='r')
        axes[1].plot(
            xs, norm(myHipLogProb.pm_ra0, myHipLogProb.pm_ra0_err).pdf(xs), 
            color='k'
        )
        axes[1].set_xlabel('PM RA [mas/yr]')

        # PM Dec
        xs = np.linspace(
            myHipLogProb.pm_dec0 - 3 * myHipLogProb.pm_dec0_err, 
            myHipLogProb.pm_dec0 + 3 * myHipLogProb.pm_dec0_err,
            1000
        )
        axes[2].hist(sampler.flatchain[:,2], bins=50, density=True, color='r')
        axes[2].plot(
            xs, norm(myHipLogProb.pm_dec0, myHipLogProb.pm_dec0_err).pdf(xs), 
            color='k'
        )
        axes[2].set_xlabel('PM Dec [mas/yr]')

        # RA offset
        axes[3].hist(sampler.flatchain[:,3], bins=50, density=True, color='r')
        xs = np.linspace(-1, 1, 1000)
        axes[3].plot(xs, norm(0, myHipLogProb.alpha0_err).pdf(xs), color='k')
        axes[3].set_xlabel('RA Offset [mas]')

        # Dec offset
        axes[4].hist(sampler.flatchain[:,4], bins=50, density=True, color='r')
        axes[4].plot(xs, norm(0, myHipLogProb.delta0_err).pdf(xs), color='k')
        axes[4].set_xlabel('Dec Offset [mas]')


        plt.tight_layout()
        plt.savefig(saveplot, dpi=250)

    return sampler.flatchain, myHipLogProb


if __name__ == '__main__':
    test_hipparcos_api()
    # _nielsen_iad_refitting_test()
