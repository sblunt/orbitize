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
    (studying the orbit of beta Pic).

    Args:
        hip_num (str): the Hipparcos number of your target. Accessible on Simbad.

    Caveats:
        Currently only treats 2-body systems. i.e. I haven't worked through the
            case where you have two massive planets that simultaneously influence the
            IAD.
    """

    def __init__(self, iad_file, hip_num):

        self.hip_num = hip_num

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

    def compute_lnprob(self, raoff_model, deoff_model, astr_samples, negative=False):
        """
        Computes the log probability of an orbit model with respect to the Hipparcos 
        IAD. 

        raoff/deoff: ra/deoffsets for star, computed for an arbitrary orbit model
        astr_samples: pm_ra, pm_dec, alpha_H0, delta_H0, plx (fitted astrometric params)

        Returns:
            np.array of length M, where M is the number of input orbits (same as def'n
                in description of `samples` arg above) representing the log probaility
                for each orbit with respect to the Hipparcos IAD
        """
        n_params = len(samples)

        # variables for each of the astrometric fitting parameters
        # TODO: check order of these params
        pm_ra = samples[0]
        pm_dec = samples[1]
        alpha_H0 = samples[2]
        delta_H0 = samples[3]
        plx = samples[4]

        else:
            print('Incorrect number of fitting params in `samples`.')
            return

        n_samples = len(pm_ra)
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
            alpha_C_st += raoff
            delta_C += decoff

            # calculate distance between line and expected measurement (Nielsen+ 2020 Eq 6) [mas]
            dist[i, :] = np.abs(
                (self.alpha_abs_st[i] - alpha_C_st) * self.cos_phi[i] + 
                (self.delta_abs[i] - delta_C) * self.sin_phi[i]
            )

        # compute chi2 (Nielsen+ 2020 Eq 7)
        chi2 = np.sum([(dist[:,i] / self.eps)**2 for i in np.arange(n_samples)], axis=1)
        lnprob = -0.5 * chi2

        return lnprob


if __name__ == '__main__':

    # instantiate an object for HR 5183
    PlanetPi = HipparcosLogProb(hip_num='027321')

    # load legacy posterior
    df_rv = pd.read_csv(
        '/data/user/lrosenth/legacy/run_final/{}/chains.csv.tar.bz2'.format(
            '120066'
        )
    )

    # subsample posterior
    n_to_sample = int(1e5)
    df_rv_subsamp = df_rv.sample(n_to_sample)

    # convert radvel posterior to orbitize basis
    _, df_orb = compute_sep(
        df_rv_subsamp, Time(np.array([0]), format='decimalyear'), 
        'per tc secosw sesinw k', 1.07, 0.04, 31.757, 0.039, 1, pl_num=1
    )

    n_samples = len(df_orb['mp'].values)

    # create an orbit model for which to calculate chi2
    pm_ra_samples = np.random.normal(PlanetPi.pm_ra0, PlanetPi.pm_ra0_err, size=n_samples)
    pm_dec_samples = np.random.normal(PlanetPi.pm_dec0, PlanetPi.pm_dec0_err, size=n_samples)
    alpha_H0_samples = np.random.normal(0, 0.1, size=n_samples)
    dec_H0_samples = np.random.normal(0, 0.1, size=n_samples)
    mtot_samples = df_orb['m_st'].values + df_orb['mp'].values

    samples = np.array([
        pm_ra_samples, pm_dec_samples, alpha_H0_samples, dec_H0_samples, 
        df_orb['plx'].values, df_orb['sma'].values, df_orb['ecc'].values,
        df_orb['inc_rad'].values, df_orb['omega_pl_rad'].values, 
        df_orb['lan_rad'].values, df_orb['tau_58849'].values, 
        mtot_samples, df_orb['mp'].values
    ])

    # compute chi2
    print('Computing lnprobs!')
    logprobs = PlanetPi.compute_lnprob(samples)

    # plot parameters for some best fitting orbits out of legacy posterior, as a sanity check
    best_orbit_indices = logprobs.argsort()[-1000:]

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))

    # semimajor axis
    ax[0].hist(samples[5], bins=50, density=True, color='grey', alpha=0.5, label='Legacy')
    ax[0].hist(
        samples[5][best_orbit_indices], bins=50, density=True, color='red', 
        histtype='step', label='best fits incl. Hip IAD'
    )
    ax[0].set_xlabel('sma [au]')
    ax[0].legend()

    # parallax
    ax[1].hist(samples[4], bins=50, density=True, color='grey', alpha=0.5)
    ax[1].hist(
        samples[4][best_orbit_indices], bins=50, density=True, color='red', 
        histtype='step'
    )
    ax[1].set_xlabel('$\pi$ [mas]')

    # ecc
    ax[2].hist(samples[6], bins=50, density=True, color='grey', alpha=0.5)
    ax[2].hist(
        samples[6][best_orbit_indices], bins=50, density=True, color='red', 
        histtype='step'
    )
    ax[2].set_xlabel('ecc')

    # pm_RA
    ax[3].hist(samples[0], bins=50, density=True, color='grey', alpha=0.5)
    ax[3].hist(
        samples[0][best_orbit_indices], bins=50, density=True, color='red', 
        histtype='step'
    )
    ax[3].set_xlabel('$\\mu_{{\\alpha}}$ [mas/yr]')

    # ra_H0
    ax[4].hist(samples[3], bins=50, density=True, color='grey', alpha=0.5)
    ax[4].hist(
        samples[3][best_orbit_indices], bins=50, density=True, color='red', 
        histtype='step'
    )
    ax[4].set_xlabel('$\\alpha_{{\\rm H0}}$ [mas]')

    plt.savefig('planetpi_test.png', dpi=250)