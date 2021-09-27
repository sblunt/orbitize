import numpy as np
from astroquery.gaia import Gaia
from astropy import units as u

class GaiaLogProb(object):

    """
    TODO: add unit tests!
    TODO: document this file
    TODO: cite Gaia appropriately
    TODO: catch errors in this code
    TODO: add Gaia to Hip tutorial
    TODO: Assumes already using Hip IAD (i.e. can't use this module by itself)-- add an error message saying that
    TODO: don't require the user to put in Gaia source ID (look it up)
    TODO: raise issue to account for correlations in Gaia measurements
    TODO: raise issue to make plots that display Hipparcos and Gaia fits
    """
    def __init__(self, gaia_num, hiplogprob, dr='dr2'): # choose from: 'dr2', 'edr3'

        # import astroquery.simbad
        # simbad = astroquery.simbad.Simbad()
        # simbad.add_votable_fields('ids')

        # hip_name = 'HIP {}'.format(hip_num)

        # df = simbad.query_objects([hip_name])

        self.hiplogprob = hiplogprob

        query = """SELECT
        TOP 1
        ra, dec, ra_error, dec_error
        FROM gaia{}.gaia_source
        WHERE source_id = {}
        """.format(dr, gaia_num)

        job = Gaia.launch_job_async(query)
        gaia_data = job.get_results()

        self.ra = gaia_data['ra']
        self.ra_err = gaia_data['ra_error']
        self.dec = gaia_data['dec']
        self.dec_err = gaia_data['dec_error']

        # keep this number on hand for use in lnlike computation 
        self.mas2deg = (u.mas).to(u.degree)

        if dr == 'edr3':
            self.gaia_epoch = 2016.0
        elif dr == 'dr2':
            self.gaia_epoch = 2015.5

        self.hipparcos_epoch = 1991.25

    def compute_lnlike(
        self, raoff_model, deoff_model, samples, param_idx
    ):
        """
        Computes the log likelihood of an orbit model with respect to the 
        Hipparcos IAD. This is added to the likelihoods calculated with 
        respect to other data types in ``sampler._logl()``. 
        Args:
            raoff_model (np.array of float): TODO
            deoff_model (np.array of float): primary RA
                offsets from the barycenter incurred from orbital motion of 
                companions (i.e. not from parallactic motion), at Gaia epoch TODO: check
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


        alpha_H0 = samples[param_idx['alpha0']] # [deg]
        pm_ra = samples[param_idx['pm_ra']] # [mas/yr]
        delta_alpha_from_pm = pm_ra * (self.gaia_epoch - self.hipparcos_epoch) # [mas]

        delta_H0 = samples[param_idx['delta0']] # [deg]
        pm_dec = samples[param_idx['pm_dec']] # [mas/yr]
        delta_delta_from_pm = pm_dec * (self.gaia_epoch - self.hipparcos_epoch) # [mas]

        # difference in position due to orbital motion between Hipparcos & Gaia epochs
        alpha_diff_orbit = (raoff_model[1,:] - raoff_model[0,:]) # [mas]
        dec_diff_orbit = (deoff_model[1,:] - deoff_model[0,:]) 

        ##### NOTE: change #1 0->1 swapped; change #2: checked all units
       
        

        # RA model (not in tangent plane)
        alpha_model = ( # [deg]
            self.hiplogprob.alpha0 + self.mas2deg * (
                alpha_H0  + 
                delta_alpha_from_pm + 
                alpha_diff_orbit

            # divide by cos(dec) to undo projection onto tangent plane
            ) / np.cos(np.radians(self.hiplogprob.delta0))  
        )
        alpha_data = self.ra

        # again divide by cos(dec) to undo projection onto tangent plane
        alpha_unc = self.mas2deg * self.ra_err / np.cos(np.radians(self.hiplogprob.delta0)) 

        # technically this is an angle so we should wrap it, but the precision
        # of Hipparcos and Gaia is so good that we'll never have to.
        alpha_resid = (alpha_model - alpha_data)
        alpha_chi2 = (alpha_resid / alpha_unc)**2

        delta_model = ( # [deg]
            self.hiplogprob.delta0 + self.mas2deg * (
                delta_H0 + 
                delta_delta_from_pm + 
                dec_diff_orbit
            )
        )
        dec_data = self.dec
        delta_unc = self.mas2deg * self.dec_err

        delta_chi2 = ((delta_model - dec_data) / delta_unc)**2

        chi2 = np.sum(alpha_chi2) + np.sum(delta_chi2)
        lnlike = -0.5 * chi2

        return lnlike


