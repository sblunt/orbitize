import numpy as np
from astropy.io import ascii
import pandas as pd
import emcee
from scipy.stats import norm
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from astroquery.vizier import Vizier

class HipparcosLogProb(object):
    """
    Class to compute the log probability of an orbit with respect to the 
    Hipparcos Intermediate Astrometric Data (IAD). If using a DVD file, 
    queries Vizier for all metadata relevant to the IAD, and reads in the IAD 
    themselves from a specified location. Follows Nielsen+ 2020 (studying the 
    orbit of beta Pic b).

    Fitting the Hipparcos IAD requires fitting for the following five parameters.
    They are added to the vector of fitting parameters in system.py, but 
    are described here for completeness. See Nielsen+ 2020 for more detail.

    - alpha0: RA offset from the reported Hipparcos position at a particular
        epoch (usually 1991.25) [mas]
    - delta0: Dec offset from the reported Hipparcos position at a particular
        epoch (usually 1991.25) [mas]
    - pm_ra: RA proper motion [mas/yr]
    - pm_dec: Dec proper motion [mas/yr]
    - plx: parallax [mas]

    .. Note:: 
    
        In orbitize, it is possible to perform a fit to just the Hipparcos
        IAD, but not to just the Gaia astrometric data.

    Args:
        path_to_iad_file (str): location of IAD file to be used in your fit.
            See the Hipparcos tutorial for a walkthrough of how to 
            download these files.
        hip_num (str): Hipparcos ID of star. Available on Simbad. Should have
            zeros in the prefix if number is <100,000. (i.e. 27321 should be
            passed in as '027321').
        num_secondary_bodies (int): number of companions in the system
        alphadec0_epoch (float): epoch (in decimal year) that the fitting 
            parameters alpha0 and delta0 are defined relative to (see above).
        renormalize_errors (bool): if True, normalize the scan errors to get
            chisq_red = 1, following Nielsen+ 2020 (eq 10). In general, this 
            should be False, but it's helpful for testing. Check out 
            `orbitize.hipparcos.nielsen_iad_refitting_test()` for an example
            using this renormalization.

    Written: Sarah Blunt & Rob de Rosa, 2021
    """

    def __init__(
        self, path_to_iad_file, hip_num, num_secondary_bodies,
        alphadec0_epoch=1991.25, renormalize_errors=False
    ):

        self.path_to_iad_file = path_to_iad_file
        self.renormalize_errors = renormalize_errors

        # infer if the IAD file is an older DVD file or a new file
        with open(path_to_iad_file, 'r') as f:
            first_char = f.readline()[0]

            # newer format files don't start with comments
            if first_char == '#':
                dvd_file = False
            else:
                dvd_file = True

        self.hip_num = hip_num
        self.num_secondary_bodies = num_secondary_bodies
        self.alphadec0_epoch = alphadec0_epoch

        # dvd files don't contain the Hipparcos astrometric solution, so
        # we need to look it up
        if dvd_file:

            # load best-fit astrometric solution from Sep 08 van Leeuwen catalog
            # (https://cdsarc.unistra.fr/ftp/I/311/ReadMe)
            Vizier.ROW_LIMIT = -1
            hip_cat = Vizier(
                catalog='I/311/hip2', 
                columns=[
                    'RArad', 'e_RArad', 'DErad', 'e_DErad', 'Plx', 'e_Plx', 'pmRA', 
                    'e_pmRA', 'pmDE', 'e_pmDE', 'F2', 'Sn'
                ]
            ).query_constraints(HIP=self.hip_num)[0]

            self.plx0 = hip_cat['Plx'][0] # [mas]
            self.pm_ra0 = hip_cat['pmRA'][0] # [mas/yr]
            self.pm_dec0 = hip_cat['pmDE'][0] # [mas/yr]
            self.alpha0 = hip_cat['RArad'][0] # [deg]
            self.delta0 = hip_cat['DErad'][0] # [deg]
            self.plx0_err = hip_cat['e_Plx'][0] # [mas]
            self.pm_ra0_err = hip_cat['e_pmRA'][0] # [mas/yr]
            self.pm_dec0_err = hip_cat['e_pmDE'][0] # [mas/yr]
            self.alpha0_err = hip_cat['e_RArad'][0] # [mas]
            self.delta0_err = hip_cat['e_DErad'][0] # [mas]

            solution_type = hip_cat['Sn'][0]
            f2 = hip_cat['F2'][0]
        
        else:
            
            # read the Hipparcos best-fit solution from the IAD file
            astrometric_solution = pd.read_csv(path_to_iad_file, skiprows=9, sep='\s+', nrows=1)
            self.plx0 = astrometric_solution['Plx'].values[0] # [mas]
            self.pm_ra0 = astrometric_solution['pm_RA'].values[0] # [mas/yr]
            self.pm_dec0 = astrometric_solution['pm_DE'].values[0] # [mas/yr]
            self.alpha0 = astrometric_solution['RAdeg'].values[0] # [deg]
            self.delta0 = astrometric_solution['DEdeg'].values[0] # [deg]
            self.plx0_err =  astrometric_solution['e_Plx'].values[0] # [mas]
            self.pm_ra0_err = astrometric_solution['e_pmRA'].values[0] # [mas/yr]
            self.pm_dec0_err = astrometric_solution['e_pmDE'].values[0] # [mas/yr]
            self.alpha0_err = astrometric_solution['e_RA'].values[0] # [mas]
            self.delta0_err = astrometric_solution['e_DE'].values[0] # [mas]

            solution_details = pd.read_csv(path_to_iad_file, skiprows=5, sep='\s+', nrows=1)

            solution_type = solution_details['isol_n'].values[0]
            f2 = solution_details['F2'].values[0]


        if solution_type != 5:
            raise ValueError(
                """
                Currently, we only handle stars with 5-parameter astrometric 
                solutions from Hipparcos. Let us know if you'd like us to add 
                functionality for stars with >5 parameter solutions.
                """
            )

        # read in IAD
        if dvd_file:
            iad = np.transpose(np.loadtxt(path_to_iad_file, skiprows=1))
        else:
            iad = np.transpose(np.loadtxt(path_to_iad_file))

        n_lines = len(iad)

        times = iad[1] + 1991.25
        self.cos_phi = iad[3] # scan direction
        self.sin_phi = iad[4]
        self.R = iad[5] # abscissa residual [mas]
        self.eps = iad[6] # error on abscissa residual [mas]

        # reject negative errors (scans that were rejected by Hipparcos team)
        good_scans = np.where(self.eps > 0)[0]

        if n_lines - len(good_scans) > 0:
            print('{} Hipparcos scans rejected.'.format(n_lines - len(good_scans)))
        times = times[good_scans]
        self.cos_phi = self.cos_phi[good_scans]
        self.sin_phi = self.sin_phi[good_scans]
        self.R = self.R[good_scans]
        self.eps = self.eps[good_scans]

        epochs = Time(times, format='decimalyear')
        self.epochs = epochs.decimalyear
        self.epochs_mjd = epochs.mjd

        if self.renormalize_errors:
            D = len(epochs) - 6
            G = f2

            f = (
                G * np.sqrt(2 / (9 * D)) + 
                1 - 
                (2 / (9 * D))
            )**(3/2)

            self.eps *= f

        # compute Earth XYZ position in barycentric coordinates
        bary_pos, _ = get_body_barycentric_posvel('earth', epochs)
        self.X = bary_pos.x.value # [au]
        self.Y = bary_pos.y.value # [au]
        self.Z = bary_pos.z.value # [au]

        # reconstruct ephemeris of star given van Leeuwen best-fit (Nielsen+ 2020 Eqs 1-2) [mas]
        changein_alpha_st = (
            self.plx0 * (
                self.X * np.sin(np.radians(self.alpha0)) - 
                self.Y * np.cos(np.radians(self.alpha0))
            ) + (self.epochs - 1991.25) * self.pm_ra0
        )

        changein_delta = (
            self.plx0 * (
                self.X * np.cos(np.radians(self.alpha0)) * np.sin(np.radians(self.delta0)) + 
                self.Y * np.sin(np.radians(self.alpha0)) * np.sin(np.radians(self.delta0)) - 
                self.Z * np.cos(np.radians(self.delta0))
            ) + (self.epochs - 1991.25) * self.pm_dec0
        )

        # compute abcissa point (Nielsen+ Eq 3)
        self.alpha_abs_st = self.R * self.cos_phi + changein_alpha_st
        self.delta_abs = self.R * self.sin_phi + changein_delta


    def _save(self, hf):
        """
        Saves the current object to an hdf5 file

        Args:
            hf (h5py._hl.files.File): a currently open hdf5 file in which
                to save the object.
        """
        with open(self.path_to_iad_file, 'r') as f:
            iad_data = np.array(f.readlines(), dtype='S')
            hf.create_dataset("IAD_datafile", data=iad_data)

        hf.attrs['hip_num'] = self.hip_num
        hf.attrs['alphadec0_epoch'] = self.alphadec0_epoch
        hf.attrs['renormalize_errors'] = self.renormalize_errors

    def compute_lnlike(
        self, raoff_model, deoff_model, samples, param_idx
    ):
        """
        Computes the log likelihood of an orbit model with respect to the 
        Hipparcos IAD. This is added to the likelihoods calculated with 
        respect to other data types in ``sampler._logl()``. 

        Args:
            raoff_model (np.array of float): M-dimensional array of primary RA
                offsets from the barycenter incurred from orbital motion of 
                companions (i.e. not from parallactic motion), where M is the 
                number of epochs of IAD scan data.
            deoff_model (np.array of float): M-dimensional array of primary RA
                offsets from the barycenter incurred from orbital motion of 
                companions (i.e. not from parallactic motion), where M is the 
                number of epochs of IAD scan data.
            samples (np.array of float): R-dimensional array of fitting 
                parameters, where R is the number of parameters being fit. Must 
                be in the same order documented in ``System``. 
            param_idx: a dictionary matching fitting parameter labels to their
                indices in an array of fitting parameters (generally 
                set to System.basis.param_idx).

        Returns:
            np.array of float: array of length M, where M is the number of input 
                orbits, representing the log likelihood of each orbit with 
                respect to the Hipparcos IAD.
        """

        # variables for each of the astrometric fitting parameters
        plx = samples[param_idx['plx']]
        pm_ra = samples[param_idx['pm_ra']]
        pm_dec = samples[param_idx['pm_dec']]
        alpha_H0 = samples[param_idx['alpha0']]
        delta_H0 = samples[param_idx['delta0']]

        try:
            n_samples = len(pm_ra)
        except TypeError:
            n_samples = 1

        n_epochs = len(self.epochs)
        dist = np.empty((n_epochs, n_samples))

        # add parallactic ellipse & proper motion to position (Nielsen+ 2020 Eq 8)
        for i in np.arange(n_epochs):

            # this is the expected offset from the photocenter in alphadec0_epoch (typically 1991.25 for Hipparcos)
            alpha_C_st = alpha_H0 + plx * (
                self.X[i] * np.sin(np.radians(self.alpha0)) - 
                self.Y[i] * np.cos(np.radians(self.alpha0))
            ) + (self.epochs[i] - self.alphadec0_epoch) * pm_ra
            delta_C = delta_H0 + plx * (
                self.X[i] * np.cos(np.radians(self.alpha0)) * np.sin(np.radians(self.delta0)) + 
                self.Y[i] * np.sin(np.radians(self.alpha0)) * np.sin(np.radians(self.delta0)) -
                self.Z[i] * np.cos(np.radians(self.delta0))
            ) + (self.epochs[i] - self.alphadec0_epoch) * pm_dec

            # add in pre-computed secondary perturbations
            alpha_C_st += raoff_model[i]
            delta_C += deoff_model[i]

            # calculate distance between line and expected measurement (Nielsen+ 2020 Eq 6) [mas]
            dist[i, :] = np.abs(
                (self.alpha_abs_st[i] - alpha_C_st) * self.cos_phi[i] + 
                (self.delta_abs[i] - delta_C) * self.sin_phi[i]
            )

        # compute chi2 (Nielsen+ 2020 Eq 7)
        chi2 = np.sum([(dist[:,i] / self.eps)**2 for i in np.arange(n_samples)], axis=1)
        lnlike = -0.5 * chi2

        return lnlike

def nielsen_iad_refitting_test(
    iad_file, hip_num='027321', saveplot='bPic_IADrefit.png', 
    burn_steps=100, mcmc_steps=5000
):
    """
    Reproduce the IAD refitting test from Nielsen+ 2020 (end of Section 3.1).
    The default MCMC parameters are what you'd want to run before using 
    the IAD for a new system. This fit uses 100 walkers. 

    Args:
        iad_loc (str): path to the IAD file.
        hip_num (str): Hipparcos ID of star. Available on Simbad. Should have
            zeros in the prefix if number is <100,000. (i.e. 27321 should be
            passed in as '027321').
        saveplot (str): what to save the summary plot as. If None, don't make a 
            plot
        burn_steps (int): number of MCMC burn-in steps to run.
        mcmc_steps (int): number of MCMC production steps to run.

    Returns:
        tuple:

            numpy.array of float: n_steps x 5 array of posterior samples
            
            orbitize.hipparcos.HipparcosLogProb: the object storing relevant
                metadata for the performed Hipparcos IAD fit
    """
    
    num_secondary_bodies = 0

    myHipLogProb = HipparcosLogProb(
        iad_file, hip_num, num_secondary_bodies, renormalize_errors=True
    )
    n_epochs = len(myHipLogProb.epochs)

    param_idx = {'plx':0, 'pm_ra':1, 'pm_dec':2, 'alpha0':3, 'delta0':4}

    def log_prob(model_pars):
        ra_model = np.zeros(n_epochs)
        dec_model = np.zeros(n_epochs)
        lnlike = myHipLogProb.compute_lnlike(
            ra_model, dec_model, model_pars, 
            param_idx
        )
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
