import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.stats import norm
from orbitize.hipparcos import HipparcosLogProb

def test_hipparcos_api():

    # check that error is caught for a star with solution type != 5 param
    hip_num = '25'
    num_secondary_bodies = 1
    iad_file = '/data/user/sblunt/HipIAD/H{}/HIP{}.d'.format(hip_num[0:3], hip_num)

    try:
        myHip = HipparcosLogProb(iad_file, hip_num, num_secondary_bodies)
    except Exception: 
        pass

def _nielsen_iad_test(hip_num='027321', saveplot='foo.png'):

    # reproduce the test from Nielsen+ 2020 (end of Section 3.1)
    
    num_secondary_bodies = 0
    iad_file = '/data/user/sblunt/HipIAD/H{}/HIP{}.d'.format(hip_num[0:3], hip_num)
    myHipLogProb = HipparcosLogProb(iad_file, hip_num, num_secondary_bodies, renormalize_errors=True)
    n_epochs = len(myHipLogProb.epochs)

    def log_prob(model_pars):
        ra_model = np.zeros(n_epochs)
        dec_model = np.zeros(n_epochs)
        lnlike = myHipLogProb.compute_lnlike(ra_model, dec_model, model_pars)
        return lnlike
    
    ndim, nwalkers = 5, 100

    # initialize walkers
    # (fitting plx, mu_a, mu_d, alpha_H0, delta_H0)
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
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    print('Starting production chain!')
    sampler.run_mcmc(state, 1000)


    if saveplot is not None:
        _, axes = plt.subplots(5, figsize=(5,12))

        # plx
        xs = np.linspace(
            myHipLogProb.plx0 - 3 * myHipLogProb.plx0_err, 
            myHipLogProb.plx0 + 3 * myHipLogProb.plx0_err,
            1000
        )
        axes[0].hist(sampler.flatchain[:,0], bins=50, density=True, color='r')
        axes[0].plot(xs, norm(myHipLogProb.plx0, myHipLogProb.plx0_err).pdf(xs), color='k')
        axes[0].set_xlabel('plx [mas]')

        # PM RA
        xs = np.linspace(
            myHipLogProb.pm_ra0 - 3 * myHipLogProb.pm_ra0_err, 
            myHipLogProb.pm_ra0 + 3 * myHipLogProb.pm_ra0_err,
            1000
        )
        axes[1].hist(sampler.flatchain[:,1], bins=50, density=True, color='r')
        axes[1].plot(xs, norm(myHipLogProb.pm_ra0, myHipLogProb.pm_ra0_err).pdf(xs), color='k')
        axes[1].set_xlabel('PM RA [mas/yr]')

        # PM Dec
        xs = np.linspace(
            myHipLogProb.pm_dec0 - 3 * myHipLogProb.pm_dec0_err, 
            myHipLogProb.pm_dec0 + 3 * myHipLogProb.pm_dec0_err,
            1000
        )
        axes[2].hist(sampler.flatchain[:,2], bins=50, density=True, color='r')
        axes[2].plot(xs, norm(myHipLogProb.pm_dec0, myHipLogProb.pm_dec0_err).pdf(xs), color='k')
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


if __name__ == '__main__':
    nielsen_iad_test()