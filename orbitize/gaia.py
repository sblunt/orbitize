import os
import numpy as np
import contextlib
import requests

with contextlib.redirect_stdout(None):
    from astroquery.gaia import Gaia
from astropy import units as u
import astropy.io.fits as fits
import astropy.time as time
from astropy.io.ascii import read
from orbitize import DATADIR
from astropy.coordinates import get_body_barycentric_posvel

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

    Written: Sarah Blunt, 2021
    """
    def __init__(self, gaia_num, hiplogprob, dr='dr2'):

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

        chi2 =   + delta_chi2
        lnlike = -0.5 * chi2 

        return lnlike



class HGCALogProb(object):
    """
    Class to compute the log probability of an orbit with respect to the proper
    motion anomalies derived jointly from Gaia and Hipparcos using the HGCA catalogs 
    produced by Brandt (2018, 2021). With this class, both Gaia and Hipparcos
    data will be considered. Do not need to pass in the Hipparcos class into System!

    Required auxiliary data:
      * HGCA of acceleration (either DR2 or EDR3)
      * Hipparcos IAD file (see orbitize.hipparcos for more info)
      * Gaia Observation Forecast Tool (GOST) CSV output (https://gaia.esac.esa.int/gost/). 

    For GOST, after entering the target name and resolving its coordinates,
    use 2014-07-26T00:00:00 as the start date. For the end date, use
    2016-05-23T00:00:00 for DR2 and 2017-05-28T00:00:00 for EDR3. 

    Args:
        hip_id (int): the Hipparcos source ID of the object you're fitting.
        hiplogprob (orbitize.hipparcos.HipLogProb): object containing
            all info relevant to Hipparcos IAD fitting
        gost_filepath (str): path to CSV file outputted by GOST
        hgca_filepath (str): path to HGCA catalog FITS file. 
            If None, will download and store in orbitize.DATADIR

    Written: Jason Wang, 2022
    """
    def __init__(self, hip_id, hiplogprob, gost_filepath, hgca_filepath=None):

        # use default HGCA catalog if not supplied
        if hgca_filepath is None:
            # check orbitize.DATAIDR and download if needed
            hgca_filepath = os.path.join(DATADIR, "HGCA_vEDR3.fits")
            if not os.path.exists(hgca_filepath):
                hgca_url = 'http://physics.ucsb.edu/~tbrandt/HGCA_vEDR3.fits'
                print("No HGCA catalog found. Downloading HGCA vEDR3 from {0} and storing into {1}.".format(hgca_url, hgca_filepath))
                hgca_file = requests.get(hgca_url)
                with open(hgca_filepath, 'wb') as f:
                    f.write(hgca_file.content)
            else:
                print("Using HGCA catalog stored in {0}".format(hgca_filepath))

        # grab the entry from the HGCA
        with fits.open(hgca_filepath) as hdulist:
            hgtable = hdulist[1].data
        entry = hgtable[np.where(hgtable['hip_id'] == hip_id)]
        # check we matched on a single target. mainly check if we typed hip id number incorrectly
        if len(entry) != 1:
            raise ValueError("HIP {0} encountered {1} matches. Expected 1 match.".format(hip_id, len(entry)))
        #  self.hgca_entry = entry

        # grav the relevant values
        hip_pm = np.array([entry['pmra_hip'][0], entry['pmdec_hip'][0]])
        hip_pm_err = np.array([entry['pmra_hip_error'][0], entry['pmdec_hip_error'][0]])

        hg_pm = np.array([entry['pmra_hg'][0], entry['pmdec_hg'][0]])
        hg_pm_err = np.array([entry['pmra_hg_error'][0], entry['pmdec_hg_error'][0]])

        gaia_pm = np.array([entry['pmra_gaia'][0], entry['pmdec_gaia'][0]])
        gaia_pm_err = np.array([entry['pmra_gaia_error'][0], entry['pmdec_gaia_error'][0]])

        # the PMa and their error bars. 
        # TODO: there are covariances to be used, but they are not being used here. 
        self.hip_hg_dpm = hip_pm - hg_pm
        self.hip_hg_dpm_err = np.sqrt(hip_pm_err**2 + hg_pm_err**2)
        self.gaia_hg_dpm = gaia_pm - hg_pm
        self.gaia_hg_dpm_err = np.sqrt(gaia_pm_err**2 + hg_pm_err**2)

        # save gaia best fit values
        self.gaia_plx0 = entry['parallax_gaia']
        self.gaia_alpha0 = entry['gaia_ra']
        self.gaia_delta0 = entry['gaia_dec']
        self.gaia_pm_ra0 = entry['pmra_gaia']
        self.gaia_pm_dec0 = entry['pmdec_gaia']

        # save the hipparcos object for use later
        #  self.hiplogprob = hiplogprob
        self.hipparcos_epoch = hiplogprob.epochs # in decimal year
        self.hipparcos_cos_phi = hiplogprob.cos_phi
        self.hipparcos_sin_phi = hiplogprob.sin_phi
        self.hipparcos_plx0 = hiplogprob.plx0
        self.hipparcos_alpha0 = hiplogprob.alpha0
        self.hipparcos_delta0 = hiplogprob.delta0
        self.hipparcos_pm_ra0 = hiplogprob.pm_ra0
        self.hipparcos_pm_dec0 = hiplogprob.pm_dec0

        # read in the GOST file to get the estimated Gaia epochs
        gost_dat = read(gost_filepath)
        self.gaia_epoch = time.Time(gost_dat["ObservationTimeAtGaia[UTC]"]).decimalyear # in decimal year
        self.gaia_scan_theta = gost_dat["scanAngle[rad]"]

        # reconstruct the model 5 parameter RA/Dec curves for both hipparcos and gaia
        # first for Hipparcos
        self.hip_bary_pos, _ = get_body_barycentric_posvel('earth', self.hipparcos_epoch)

        # reconstruct ephemeris of star given van Leeuwen best-fit (Nielsen+ 2020 Eqs 1-2) [mas]
        self.hip_changein_alpha_st = (
            self.hipparcos_plx0 * (
                self.hip_bary_pos.x.value * np.sin(np.radians(self.hipparcos_alpha0)) - 
                self.hip_bary_pos.y.value * np.cos(np.radians(self.hipparcos_alpha0))
            ) + (self.hipparcos_epoch - 1991.25) * self.hipparcos_pm_ra0
        )
        self.hip_changein_delta = (
            hiplogprob.plx0 * (
                self.hip_bary_pos.x.value * np.cos(np.radians(self.hipparcos_alpha0)) * np.sin(np.radians(self.hipparcos_delta0)) + 
                self.hip_bary_pos.y.value * np.sin(np.radians(self.hipparcos_alpha0)) * np.sin(np.radians(self.hipparcos_delta0)) - 
                self.hip_bary_pos.z.value * np.cos(np.radians(self.hipparcos_delta0))
            ) + (self.hipparcos_epoch - 1991.25) * self.hipparcos_pm_dec0
        )

        # now for Gaia
        self.gaia_bary_pos, _ = get_body_barycentric_posvel('earth', self.gaia_epoch)

        self.gaia_changein_alpha_st = (
            self.gaia_plx0 * (
                self.gaia_bary_pos.x.value * np.sin(np.radians(self.gaia_alpha0)) - 
                self.gaia_bary_pos.y.value * np.cos(np.radians(self.gaia_alpha0))
            ) + (self.gaia_epoch - entry['epoch_ra_gaia']) * self.gaia_pm_ra0
        )

        self.gaia_changein_delta = (
            self.gaia_plx0 * (
                self.gaia_bary_pos.x.value * np.cos(np.radians(self.gaia_alpha0)) * np.sin(np.radians(self.gaia_delta0)) + 
                self.gaia_bary_pos.y.value * np.sin(np.radians(self.gaia_alpha0)) * np.sin(np.radians(self.gaia_delta0)) - 
                self.gaia_bary_pos.z.value * np.cos(np.radians(self.gaia_delta0))
            ) + (self.gaia_epoch - entry['epoch_dec_gaia']) * self.gaia_pm_dec0
        )
    
    def _save(self, hf):
        """
        Saves the current object to an hdf5 file

        Args:
            hf (h5py._hl.files.File): a currently open hdf5 file in which
                to save the object.
        """
        # TODO: save stuff here if needed
        #  self.hiplogprob._save(hf)
        pass

    def compute_lnlike(
        self, raoff_model, deoff_model, samples, param_idx
    ):
        """
        Computes the log likelihood of an orbit model with respect to a single 
        Gaia astrometric point. This is added to the likelihoods calculated with 
        respect to other data types in ``sampler._logl()``. 

        Args:
            raoff_model (np.array of float): NxM primary RA
                offsets from the barycenter incurred from orbital motion of 
                companions (i.e. not from parallactic motion), where N is the
                number of epochs of combined from Hipparcos and Gaia and M is the 
                number of orbits being tested, and raoff_model[0,:] are position
                predictions at the Hipparcos epoch, and raoff_model[1,:] are
                position predictions at the Gaia epoch
            deoff_model (np.array of float): NxM primary decl
                offsets from the barycenter incurred from orbital motion of 
                companions (i.e. not from parallactic motion), where N is the
                number of epochs of combined from Hipparcos and Gaia and M is the  
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
        # find the split between hipparcos and gaia data
        gaia_index = len(self.hipparcos_epoch)

        # Begin forward modeling the data of the HGCA: linear motion during the Hip
        # and Gaia missions, and the linear motion of the star between the two missions 
        plx = samples[param_idx['plx']]

        # fit linear motion in RA/Dec to the star during the Hipparcos epoch
        model_ra_hip = raoff_model[:gaia_index]
        # model_hip_pmra = np.polyfit(self.hipparcos_epoch, model_ra_hip, 1)[0,0] # mas/yr (get slope from polyfit)
        model_dec_hip = deoff_model[:gaia_index]
        
        def optimize_pm(fitparams):
            guess_pm_ra, guess_pm_dec = fitparams
            guess_hip_changein_alpha_st = (
                plx * (
                    self.hip_bary_pos.x.value * np.sin(np.radians(self.hipparcos_alpha0)) - 
                    self.hip_bary_pos.y.value * np.cos(np.radians(self.hipparcos_alpha0))
                ) + (self.hipparcos_epoch - 1991.25) * guess_pm_ra
            )
            guess_hip_changein_delta = (
                plx * (
                    self.hip_bary_pos.x.value * np.cos(np.radians(self.hipparcos_alpha0)) * np.sin(np.radians(self.hipparcos_delta0)) + 
                    self.hip_bary_pos.y.value * np.sin(np.radians(self.hipparcos_alpha0)) * np.sin(np.radians(self.hipparcos_delta0)) - 
                    self.hip_bary_pos.z.value * np.cos(np.radians(self.hipparcos_delta0))
                ) + (self.hipparcos_epoch - 1991.25) * guess_pm_dec
            )

            guess_hip_changein_alpha_st += model_ra_hip
            guess_hip_changein_delta += model_dec_hip



        
        
        model_hip_pm = np.array([model_hip_pmra, model_hip_pmdec])

        # fit linear motion in RA/Dec to the star in Gaia epoch
        model_ra_gaia = raoff_model[gaia_index:]
        model_gaia_pmra = np.polyfit(self.gaia_epoch, model_ra_gaia, 1)[0,0]  # mas/yr
        model_dec_gaia = deoff_model[gaia_index:]
        model_gaia_pmdec = np.polyfit(self.gaia_epoch, model_dec_gaia, 1)[0,0] # mas/yr
        model_gaia_pm = np.array([model_gaia_pmra, model_gaia_pmdec])

        # compute the PM difference betwen Hipparcos and Gaia positions. 
        hg_dt = np.mean(self.gaia_epoch) - np.mean(self.hipparcos_epoch)
        model_hg_pmra = (np.mean(model_ra_gaia) - np.mean(model_ra_hip))/hg_dt
        model_hg_pmdec = (np.mean(model_dec_gaia) - np.mean(model_dec_hip))/hg_dt
        model_hg_pm = np.array([model_hg_pmra, model_hg_pmdec]) 

        # take the difference between the linear motion measured during a mission, and the
        # linear motion of the star between missions. These are our observables we compare
        # to the data. Each variable contains both RA and Dec. 
        model_hip_hg_dpm = model_hip_pm - model_hg_pm
        model_gaia_hg_dpm = model_gaia_pm - model_hg_pm
        
        chi2 = (model_hip_hg_dpm - self.hip_hg_dpm)**2/(self.hip_hg_dpm_err)**2
        chi2 += (model_gaia_hg_dpm - self.gaia_hg_dpm)**2/(self.gaia_hg_dpm_err)**2
        lnlike = -0.5 * np.sum(chi2)

        return lnlike