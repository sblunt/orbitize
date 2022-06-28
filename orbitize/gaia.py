import numpy as np
import contextlib

with contextlib.redirect_stdout(None):
    from astroquery.gaia import Gaia
from astropy import units as u

class GaiaLogProb(object):
    """
    Class to compute the log probability of an orbit with respect to a single 
    astrometric position point from Gaia. Uses astroquery to look up Gaia
    astrometric data, and computes log-likelihood. To be used in conjunction with
    orbitize.hipparcos.HipLogProb; see documentation for that object for more
    detail. 

    Follows Nielsen+ 2020 (studying the orbit of beta Pic b). Note that this
    class currently only fits for the position of the star in the Gaia epoch,
    not the star's proper motion.

    .. Note:: 
    
        In orbitize, it is possible to perform a fit to just the Hipparcos
        IAD, but not to just the Gaia astrometric data.

    Args:
        gaia_num (int): the Gaia source ID of the object you're fitting. Note
            that the dr2 and edr3 source IDs are not necessarily the same.
        hiplogprob (orbitize.hipparcos.HipLogProb): object containing
            all info relevant to Hipparcos IAD fitting
        dr (str): either 'dr2' or 'edr3'
        query (bool): if True, queries the Gaia database for astrometry of the
            target (requires an internet connection). If False, uses user-input 
            astrometric values (runs without internet).
        gaia_data (dict): see `query` keyword above. If `query` set to False, 
            then user must supply a dictionary of Gaia astometry in the following
            form:
                gaia_data = {
                    'ra': 139.4 # RA in degrees
                    'dec': 139.4 # Dec in degrees
                    'ra_error': 0.004 # RA error in mas
                    'dec_error': 0.004 # Dec error in mas
                }

    Written: Sarah Blunt, 2021
    """
    def __init__(self, gaia_num, hiplogprob, dr='dr2', query=True, gaia_data=None):

        self.gaia_num = gaia_num
        self.hiplogprob = hiplogprob
        self.dr = dr

        if self.dr == 'edr3':
            self.gaia_epoch = 2016.0
        elif self.dr == 'dr2':
            self.gaia_epoch = 2015.5
        else:
            raise ValueError("`dr` must be either `dr2` or `edr3`")
        self.hipparcos_epoch = 1991.25


        if query:
            query = """SELECT
            TOP 1
            ra, dec, ra_error, dec_error
            FROM gaia{}.gaia_source
            WHERE source_id = {}
            """.format(self.dr, self.gaia_num)

            job = Gaia.launch_job_async(query)
            gaia_data = job.get_results()

        self.ra = gaia_data['ra']
        self.ra_err = gaia_data['ra_error']
        self.dec = gaia_data['dec']
        self.dec_err = gaia_data['dec_error']

        # keep this number on hand for use in lnlike computation 
        self.mas2deg = (u.mas).to(u.degree)
    
    def _save(self, hf):
        """
        Saves the current object to an hdf5 file

        Args:
            hf (h5py._hl.files.File): a currently open hdf5 file in which
                to save the object.
        """
        hf.attrs['gaia_num'] = self.gaia_num
        hf.attrs['dr'] = self.dr
        self.hiplogprob._save(hf)

    def compute_lnlike(
        self, raoff_model, deoff_model, samples, param_idx
    ):
        """
        Computes the log likelihood of an orbit model with respect to a single 
        Gaia astrometric point. This is added to the likelihoods calculated with 
        respect to other data types in ``sampler._logl()``. 

        Args:
            raoff_model (np.array of float): 2xM primary RA
                offsets from the barycenter incurred from orbital motion of 
                companions (i.e. not from parallactic motion), where M is the 
                number of orbits being tested, and raoff_model[0,:] are position
                predictions at the Hipparcos epoch, and raoff_model[1,:] are
                position predictions at the Gaia epoch
            deoff_model (np.array of float): 2xM primary decl
                offsets from the barycenter incurred from orbital motion of 
                companions (i.e. not from parallactic motion), where M is the 
                number of orbits being tested, and deoff_model[0,:] are position
                predictions at the Hipparcos epoch, and deoff_model[1,:] are
                position predictions at the Gaia epoch
            samples (np.array of float): R-dimensional array of fitting 
                parameters, where R is the number of parameters being fit. Must 
                be in the same order documented in ``System``. 
            param_idx: a dictionary matching fitting parameter labels to their
                indices in an array of fitting parameters (generally 
                set to System.basis.param_idx).
                
        Returns:
            np.array of float: array of length M, where M is the number of input 
                orbits, representing the log likelihood of each orbit with 
                respect to the Gaia position measurement.
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

        chi2 = alpha_chi2 + delta_chi2
        lnlike = -0.5 * chi2 

        return lnlike


