import numpy as np
from astroquery.gaia import Gaia

class GaiaLogProb(object):

    """
    TODO: cite Gaia appropriately
    TODO: catch errors in this code
    TODO: don't require the user to put in Gaia source ID (look it up)
    TODO: account for correlations in Gaia measurements
    TODO: make plots that display Hipparcos and Gaia
    TODO: average over Gaia motion to compare PM (compute PM anomaly)

    Assumes already using Hip IAD (i.e. can't use this module by itself)-- add an error message saying that

    """
    def __init__(self, gaia_edr3_num, hiplogprob):

        # import astroquery.simbad
        # simbad = astroquery.simbad.Simbad()
        # simbad.add_votable_fields('ids')

        # hip_name = 'HIP {}'.format(hip_num)

        # df = simbad.query_objects([hip_name])

        self.hiplogprob = hiplogprob

        query = """SELECT
        TOP 1
        ra, dec, ra_error, dec_error
        FROM gaiaedr3.gaia_source
        WHERE source_id = {}
        """.format(gaia_edr3_num)

        job = Gaia.launch_job_async(query)
        gaia_data = job.get_results()

        self.ra = gaia_data['ra']
        self.ra_err = gaia_data['ra_error']
        self.dec = gaia_data['dec']
        self.dec_err = gaia_data['dec_error']

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

        # compare alpha difference, delta difference plus orbit motion to 
        # get orbital motion 

        alpha_H0 = samples[param_idx['alpha0']]
        pm_ra = samples[param_idx['pm_ra']]
        delta_alpha_from_pm = pm_ra * (2015.5 - 1991.25)

        delta_H0 = samples[param_idx['delta0']]
        pm_dec = samples[param_idx['pm_dec']]
        delta_delta_from_pm = pm_dec * (2015.5 - 1991.25)

        alpha_model = (
            (self.ra - (self.hiplogprob.alpha0 + alpha_H0)) * np.cos(self.hiplogprob.delta0) + 
            delta_alpha_from_pm + 
            raoff_model
        )
        alpha_data = (self.ra - self.hiplogprob.alpha0) * np.cos(self.hiplogprob.delta0)
        alpha_unc = self.ra_err

        alpha_chi2 = ((alpha_model - alpha_data) / alpha_unc)**2

        delta_model = (
            (self.dec - (self.hiplogprob.delta0 + delta_H0)) + 
            delta_delta_from_pm + 
            deoff_model
        )
        dec_data = (self.dec - self.hiplogprob.delta0)
        delta_unc = self.dec_err

        delta_chi2 = ((delta_model - dec_data) / delta_unc)**2

        chi2 = np.sum(alpha_chi2) + np.sum(delta_chi2)
        lnlike = -0.5 * chi2

        return lnlike


