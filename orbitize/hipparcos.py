import numpy as np
import pandas as pd

from orbitize.kepler import calc_orbit
from orbitize.radvel_utils.compute_sep import compute_sep
from astroquery.vizier import Vizier
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel

class HipparcosLogProb(object):
    """
    Class to compute the log probability of an orbit with respect to the 
    Hipparcos Intermediate Astrometric Data (IAD). Queries Vizier for 
    all metadata relevant to the IAD, and reads in the IAD themselves from
    a specified location. Follows Nielsen+ 2020
    (studying the orbit of beta Pic b).

    Args:
        iad_file (str): 
        hip_num (str): the Hipparcos number of your target. Accessible on Simbad.
    """

    def __init__(self, iad_file, hip_num, num_secondary_bodies):

        self.hip_num = hip_num
        self.num_secondary_bodies = num_secondary_bodies

        # load best-fit astrometric solution from van Leeuwen catalog
        Vizier.ROW_LIMIT = -1
        hip_cat = Vizier(
            catalog='I/311/hip2', 
            columns=[
                'RArad', 'e_RArad', 'DErad', 'e_DErad', 'Plx', 'e_Plx', 'pmRA', 
                'e_pmRA', 'pmDE', 'e_pmDE', 'F2'
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

        # read in IAD
        iad = np.transpose(np.loadtxt(iad_file, skiprows=1))

        times = iad[1] + 1991.25
        epochs = Time(times, format='decimalyear')
        self.epochs = epochs.decimalyear
        self.epochs_mjd = epochs.mjd
        self.cos_phi = iad[3] # scan direction
        self.sin_phi = iad[4]
        self.R = iad[5] # abscissa residual [mas]
        self.eps = iad[6] # error on abscissa residual [mas]

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

    def compute_lnlike(self, raoff_model, deoff_model, samples, negative=False):
        """
        Computes the log probability of an orbit model with respect to the Hipparcos 
        IAD. 

        raoff/deoff: ra/deoffsets for star, computed for an arbitrary orbit model
        astr_samples: pm_ra, pm_dec, alpha_H0, delta_H0, plx (fitted astrometric params)

        Returns:
            np.array of length M, where M is the number of input orbits (same as def'n
                in description of `samples` arg above) representing the log probability
                for each orbit with respect to the Hipparcos IAD
        """
        n_params = len(samples)

        # variables for each of the astrometric fitting parameters
        plx = samples[6 * self.num_secondary_bodies]
        pm_ra = samples[6 * self.num_secondary_bodies + 1]
        pm_dec = samples[6 * self.num_secondary_bodies + 2]
        alpha_H0 = samples[6 * self.num_secondary_bodies + 3]
        delta_H0 = samples[6 * self.num_secondary_bodies + 4]


        try:
            n_samples = len(pm_ra)
        except TypeError:
            n_samples = 1

        n_epochs = len(self.epochs)
        dist = np.empty((n_epochs, n_samples))

        # add parallactic ellipse & proper motion to position (Nielsen+ 2020 Eq 8)
        for i in np.arange(n_epochs):

            # this is the expected offset from the Hipparcos photocenter in 1991.25
            alpha_C_st = alpha_H0 + plx * (
                self.X[i] * np.sin(np.radians(self.alpha0)) - 
                self.Y[i] * np.cos(np.radians(self.alpha0))
            ) + (self.epochs[i] - 1991.25) * pm_ra
            delta_C = delta_H0 + plx * (
                self.X[i] * np.cos(np.radians(self.alpha0)) * np.sin(np.radians(self.delta0)) + 
                self.Y[i] * np.sin(np.radians(self.alpha0)) * np.sin(np.radians(self.delta0)) -
                self.Z[i] * np.cos(np.radians(self.delta0))
            ) + (self.epochs[i] - 1991.25) * pm_dec

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