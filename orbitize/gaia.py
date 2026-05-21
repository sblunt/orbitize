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
from astropy.coordinates import get_body_barycentric_posvel
import numpy.linalg

from orbitize import DATADIR
import orbitize.lnlike


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

    def __init__(self, gaia_num, hiplogprob, dr="dr2", query=True, gaia_data=None):
        self.gaia_num = gaia_num
        self.hiplogprob = hiplogprob
        self.dr = dr

        if self.dr == "edr3":
            self.gaia_epoch = 2016.0
        elif self.dr == "dr2":
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
            """.format(
                self.dr, self.gaia_num
            )

            job = Gaia.launch_job_async(query)
            gaia_data = job.get_results()

        self.ra = gaia_data["ra"]
        self.ra_err = gaia_data["ra_error"]
        self.dec = gaia_data["dec"]
        self.dec_err = gaia_data["dec_error"]

        # keep this number on hand for use in lnlike computation
        self.mas2deg = (u.mas).to(u.degree)

    def _save(self, hf):
        """
        Saves the current object to an hdf5 file

        Args:
            hf (h5py._hl.files.File): a currently open hdf5 file in which
                to save the object.
        """
        hf.attrs["gaia_num"] = self.gaia_num
        hf.attrs["dr"] = self.dr
        self.hiplogprob._save(hf)

    def compute_lnlike(self, raoff_model, deoff_model, samples, param_idx):
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

        alpha_H0 = samples[param_idx["alpha0"]]  # [deg]
        pm_ra = samples[param_idx["pm_ra"]]  # [mas/yr]
        delta_alpha_from_pm = pm_ra * (self.gaia_epoch - self.hipparcos_epoch)  # [mas]

        delta_H0 = samples[param_idx["delta0"]]  # [deg]
        pm_dec = samples[param_idx["pm_dec"]]  # [mas/yr]
        delta_delta_from_pm = pm_dec * (self.gaia_epoch - self.hipparcos_epoch)  # [mas]

        # difference in position due to orbital motion between Hipparcos & Gaia epochs
        alpha_diff_orbit = raoff_model[1, :] - raoff_model[0, :]  # [mas]
        dec_diff_orbit = deoff_model[1, :] - deoff_model[0, :]

        # RA model (not in tangent plane)
        alpha_model = self.hiplogprob.alpha0 + self.mas2deg * (  # [deg]
            alpha_H0
            + delta_alpha_from_pm
            + alpha_diff_orbit
            # divide by cos(dec) to undo projection onto tangent plane
        ) / np.cos(np.radians(self.hiplogprob.delta0))
        alpha_data = self.ra

        # again divide by cos(dec) to undo projection onto tangent plane
        alpha_unc = (
            self.mas2deg * self.ra_err / np.cos(np.radians(self.hiplogprob.delta0))
        )

        # technically this is an angle so we should wrap it, but the precision
        # of Hipparcos and Gaia is so good that we'll never have to.
        alpha_resid = alpha_model - alpha_data
        alpha_chi2 = (alpha_resid / alpha_unc) ** 2

        delta_model = self.hiplogprob.delta0 + self.mas2deg * (  # [deg]
            delta_H0 + delta_delta_from_pm + dec_diff_orbit
        )
        dec_data = self.dec
        delta_unc = self.mas2deg * self.dec_err

        delta_chi2 = ((delta_model - dec_data) / delta_unc) ** 2

        chi2 = alpha_chi2 + delta_chi2
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
                hgca_url = (
                    "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/42/HGCA_vEDR3.fits"
                )
                print(
                    "No HGCA catalog found. Downloading HGCA vEDR3 from {0} and storing into {1}.".format(
                        hgca_url, hgca_filepath
                    )
                )
                hgca_file = requests.get(hgca_url, verify=False)
                with open(hgca_filepath, "wb") as f:
                    f.write(hgca_file.content)
            else:
                print("Using HGCA catalog stored in {0}".format(hgca_filepath))

        # grab the entry from the HGCA
        with fits.open(
            hgca_filepath, ignore_missing_simple=True, ignore_missing_end=True
        ) as hdulist:
            hgtable = hdulist[1].data
        entry = hgtable[np.where(hgtable["hip_id"] == hip_id)]
        # check we matched on a single target. mainly check if we typed hip id number incorrectly
        if len(entry) != 1:
            raise ValueError(
                "HIP {0} encountered {1} matches. Expected 1 match.".format(
                    hip_id, len(entry)
                )
            )
        #  self.hgca_entry = entry
        self.hip_id = hip_id

        # grab the relevant PM and uncertainties from HGCA
        self.hip_pm = np.array([entry["pmra_hip"][0], entry["pmdec_hip"][0]])
        self.hip_pm_err = np.array(
            [entry["pmra_hip_error"][0], entry["pmdec_hip_error"][0]]
        )
        hip_radec_cov = (
            entry["pmra_pmdec_hip"][0]
            * entry["pmra_hip_error"][0]
            * entry["pmdec_hip_error"][0]
        )

        self.hg_pm = np.array([entry["pmra_hg"][0], entry["pmdec_hg"][0]])
        self.hg_pm_err = np.array(
            [entry["pmra_hg_error"][0], entry["pmdec_hg_error"][0]]
        )
        hg_radec_cov = (
            entry["pmra_pmdec_hg"][0]
            * entry["pmra_hg_error"][0]
            * entry["pmdec_hg_error"][0]
        )

        self.gaia_pm = np.array([entry["pmra_gaia"][0], entry["pmdec_gaia"][0]])
        self.gaia_pm_err = np.array(
            [entry["pmra_gaia_error"][0], entry["pmdec_gaia_error"][0]]
        )
        gaia_radec_cov = (
            entry["pmra_pmdec_gaia"][0]
            * entry["pmra_gaia_error"][0]
            * entry["pmdec_gaia_error"][0]
        )

        # compute the differential PMs by subtracting Hip and Gaia from HG. Also propogate errors
        self.hip_hg_dpm = self.hip_pm - self.hg_pm
        self.hip_hg_dpm_err = np.sqrt(self.hip_pm_err**2 + self.hg_pm_err**2)
        self.hip_hg_dpm_radec_corr = (hip_radec_cov + hg_radec_cov) / (
            self.hip_hg_dpm_err[0] * self.hip_hg_dpm_err[1]
        )
        self.gaia_hg_dpm = self.gaia_pm - self.hg_pm
        self.gaia_hg_dpm_err = np.sqrt(self.gaia_pm_err**2 + self.hg_pm_err**2)
        self.gaia_hg_dpm_radec_corr = (gaia_radec_cov + hg_radec_cov) / (
            self.gaia_hg_dpm_err[0] * self.gaia_hg_dpm_err[1]
        )

        # condense together into orbitize "observations"
        self.dpm_data = np.array([self.hip_hg_dpm, self.gaia_hg_dpm])
        self.dpm_err = np.array([self.hip_hg_dpm_err, self.gaia_hg_dpm_err])
        self.dpm_corr = np.array(
            [self.hip_hg_dpm_radec_corr, self.gaia_hg_dpm_radec_corr]
        )

        # grab reference epochs for Gaia from HGCA so we can forward model it
        self.gaia_epoch_ra = entry["epoch_ra_gaia"][0]
        self.gaia_epoch_dec = entry["epoch_dec_gaia"][0]
        # read in the GOST file to get the estimated Gaia epochs and scan angles
        gost_dat = read(gost_filepath, converters={"*": [int, float, bytes]})
        self.gaia_epoch = time.Time(
            gost_dat["ObservationTimeAtGaia[UTC]"]
        ).decimalyear  # in decimal year
        gaia_scan_theta = np.array(gost_dat["scanAngle[rad]"])
        gaia_scan_phi = gaia_scan_theta + np.pi / 2
        self.gaia_cos_phi = np.cos(gaia_scan_phi)
        self.gaia_sin_phi = np.sin(gaia_scan_phi)
        self.gost_filepath = gost_filepath  # save for saving

        # save the same infor from Hipparcos. we also have the errors on the Hipparcos data
        self.hipparcos_epoch = hiplogprob.epochs  # in decimal year
        self.hipparcos_cos_phi = hiplogprob.cos_phi
        self.hipparcos_sin_phi = hiplogprob.sin_phi
        self.hipparcos_epoch_ra = entry["epoch_ra_hip"][0]
        self.hipparcos_epoch_dec = entry["epoch_dec_hip"][0]
        self.hippaarcos_errs = hiplogprob.eps
        self.hiplogprob = hiplogprob  # save for saving

    def _save(self, hf):
        """
        Saves the current object to an hdf5 file

        Args:
            hf (h5py._hl.files.File): a currently open hdf5 file in which
                to save the object.
        """
        # save hipparocs IAD using exiting code
        self.hiplogprob._save(hf)

        # save Gaia GOST file. avoid unicode!!
        gost_dat = read(self.gost_filepath, converters={"*": [int, float, bytes]})
        hf.create_dataset("Gaia_GOST", data=gost_dat)

    def compute_lnlike(self, raoff_model, deoff_model, samples, param_idx):
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
        # for now, think about only 1 model at a time to not break our brains
        model_ra_hip = raoff_model[:gaia_index, 0]
        model_dec_hip = deoff_model[:gaia_index, 0]
        model_ra_gaia = raoff_model[gaia_index:, 0]
        model_dec_gaia = deoff_model[gaia_index:, 0]

        hip_fit = self._linear_pm_fit(
            self.hipparcos_epoch,
            model_ra_hip,
            model_dec_hip,
            self.hipparcos_epoch_ra,
            self.hipparcos_epoch_dec,
            self.hipparcos_sin_phi,
            self.hipparcos_cos_phi,
            self.hippaarcos_errs,
        )
        model_hip_pos_offset = hip_fit[0:2]
        model_hip_pm = hip_fit[2:4]

        gaia_fit = self._linear_pm_fit(
            self.gaia_epoch,
            model_ra_gaia,
            model_dec_gaia,
            self.gaia_epoch_ra,
            self.gaia_epoch_dec,
            self.gaia_sin_phi,
            self.gaia_cos_phi,
            1,
        )
        model_gaia_pos_offset = gaia_fit[0:2]
        model_gaia_pm = gaia_fit[2:4]

        # compute the change in position between hipparcos and gaia
        hg_dalpha_st = model_gaia_pos_offset[0] - model_hip_pos_offset[0]
        model_hg_pmra = hg_dalpha_st / (self.gaia_epoch_ra - self.hipparcos_epoch_ra)

        hg_ddelta = model_gaia_pos_offset[1] - model_hip_pos_offset[1]
        model_hg_pmdec = hg_ddelta / (self.gaia_epoch_dec - self.hipparcos_epoch_dec)
        model_hg_pm = np.array([model_hg_pmra, model_hg_pmdec])

        # take the difference between the linear motion measured during a mission, and the
        # linear motion of the star between missions. These are our observables we compare
        # to the data. Each variable contains both RA and Dec.
        model_hip_hg_dpm = model_hip_pm - model_hg_pm
        model_gaia_hg_dpm = model_gaia_pm - model_hg_pm

        # chi2 = (model_hip_hg_dpm - self.hip_hg_dpm)**2/(self.hip_hg_dpm_err)**2
        # chi2 += (model_gaia_hg_dpm - self.gaia_hg_dpm)**2/(self.gaia_hg_dpm_err)**2

        lnlike = orbitize.lnlike.chi2_lnlike(
            self.dpm_data,
            self.dpm_err,
            self.dpm_corr,
            np.array([model_hip_hg_dpm, model_gaia_hg_dpm]),
            np.zeros(self.dpm_data.shape),
            [],
        )

        return np.sum(lnlike)

    def _linear_pm_fit(
        self,
        epochs,
        raoff_planet,
        decoff_planet,
        epoch_ref_ra,
        epoch_ref_dec,
        sin_phi,
        cos_phi,
        errs,
    ):
        """
        Performs a linear leastsq fit to determine the inferred proper motion given the stellar orbit around the barycenter
        """
        # Sovle y = A * x
        # construct A matrix
        A_pmra = cos_phi * (epochs - epoch_ref_ra) / errs
        A_raoff = cos_phi / errs
        A_pmdec = sin_phi * (epochs - epoch_ref_dec) / errs
        A_decoff = sin_phi / errs
        A_matrix = np.vstack((A_raoff, A_decoff, A_pmra, A_pmdec)).T

        # construct y matrix
        y_vec = (raoff_planet * cos_phi + decoff_planet * sin_phi) / errs

        x, res, _, _ = numpy.linalg.lstsq(A_matrix, y_vec, rcond=None)

        return x


import numpy as np
import astropy.time as time
from astropy.io.ascii import read

from orbitize import priors


class DR4LogProb(object):
    """
    Class to compute the log probability of an orbit with respect to Gaia DR4
    astrometric measurements (epoch astrometry).
    We treat the four linear astrometric parameters (proper motion RA/Dec and position RA/Dec)
    due to linear motion
    as explicit MCMC parameters constrained
    by Gaussian priors from the Gaia catalog.

    The forward model for each scan at epoch t_i is::

        eta_i = dr4_ra_off  * sin(psi_i)  +  dr4_dec_off * cos(psi_i)
              + dr4_pmra * dt_i * sin(psi_i)  +  dr4_pmdec * dt_i * cos(psi_i)
              + plx * parallax_factor_i
              + raoff_orbit_i * sin(psi_i)  +  deoff_orbit_i * cos(psi_i)

    where psi_i is the scan position angle, dt_i = (t_i - t_ref) in Julian
    years, and the last line is the orbital perturbation of the primary star
    passed in by the sampler.

    Required parameters in the MCMC state vector (in addition to the usual
    orbital elements, plx, masses):

        dr4_ra_off   – positional offset in RA* at reference epoch [mas]
        dr4_dec_off  – positional offset in Dec  at reference epoch [mas]
        dr4_pmra     – proper motion in RA*  [mas/yr]
        dr4_pmdec    – proper motion in Dec  [mas/yr]

    These are registered automatically when the System is constructed with
    gaia=dr4 (requires the corresponding hooks in system.py; see
    extra_param_names and extra_param_priors below). Alternatively,
    they can be injected manually after System creation.

    Pass this object into the gaia keyword of orbitize.system.System.
    You must 'set fit_secondary_mass=True' so that the star's barycentric
    wobble is computed.

    Args:
        gaia_num (int): Gaia source ID
        dr4_filepath (str): path to CSV file containing DR4 epoch astrometry
            with columns: 'obs_time_tcb', 'centroid_pos_al',
            'centroid_pos_error_al', 'parallax_factor_al',
            'scan_pos_angle', 'field_of_view'
        ref_epoch_jd (float): reference epoch in JD (TCB) for the linear
            astrometric model.  Default 2457936.875 (J2017.5).
        catalog_pmra (float): Gaia catalog proper motion in RA* [mas/yr].
        catalog_pmra_err (float): 1-sigma uncertainty on catalog_pmra.
        catalog_pmdec (float): Gaia catalog proper motion in Dec [mas/yr].
        catalog_pmdec_err (float): 1-sigma uncertainty on catalog_pmdec.
        catalog_ra_off (float): positional offset in RA* at ref_epoch_jd [mas].  Default 0.
        catalog_ra_off_err (float): 1-sigma uncertainty [mas].  Default 1 (weakly informative; Gaia positions are sub-mas,
         but the offset is defined relative to an arbitrary origin).
        catalog_dec_off (float): positional offset in Dec at `ef_epoch_jd [mas].  Default 0.
        catalog_dec_off_err (float): 1-sigma uncertainty [mas].  Default 1.

    Written: Clarissa Do O, 2026
    """

    # The four parameter names this class injects into the System.
    extra_param_names = ("dr4_ra_off", "dr4_dec_off", "dr4_pmra", "dr4_pmdec")

    def __init__(
        self,
        gaia_num,
        dr4_filepath,
        ref_epoch_jd=2457936.875, #2017.5, same as tutorial
        catalog_pmra=0.0,
        catalog_pmra_err=100.0,
        catalog_pmdec=0.0,
        catalog_pmdec_err=100.0,
        catalog_ra_off=0.0,
        catalog_ra_off_err=1.0,
        catalog_dec_off=0.0,
        catalog_dec_off_err=1.0,
    ):
        self.gaia_num = gaia_num
        self.dr4_filepath = dr4_filepath
        self.ref_epoch_jd = ref_epoch_jd

        # read DR4 epoch astrometry
        dr4_dat = read(dr4_filepath)

        epochs_time = time.Time(
            dr4_dat["obs_time_tcb"], format="jd", scale="tcb"
        )
        epochs_decyr = epochs_time.decimalyear
        epochs_jd = np.array(dr4_dat["obs_time_tcb"])

        self.n_obs = len(epochs_decyr)

        # along scan measurements and errors [mas]
        self.centroid_pos_al = np.array(dr4_dat["centroid_pos_al"])
        self.centroid_pos_error_al = np.array(dr4_dat["centroid_pos_error_al"])

        # parallax factors (pre-projected onto scan direction)
        self.parallax_factor_al = np.array(dr4_dat["parallax_factor_al"])

        # scan position angles [rad] and their sin/cos
        scan_angle = np.array(dr4_dat["scan_pos_angle"])
        self.sin_scan = np.sin(scan_angle)
        self.cos_scan = np.cos(scan_angle)

        # FOV identifier, currently not used but read in.
        self.fov = np.array(dr4_dat["field_of_view"])

        # time offsets from reference epoch [Julian yr]
        self.dt_yr = (epochs_jd - ref_epoch_jd) / 365.25

        # pre-compute scan projection of PM basis vectors
        self.pm_ra_basis = self.dt_yr * self.sin_scan   # pmra, projected
        self.pm_dec_basis = self.dt_yr * self.cos_scan   # pmdec, projected

        # constant part of the lnlike normalization
        self._log_norm = np.sum(np.log(2.0 * np.pi * self.centroid_pos_error_al ** 2))
        self._weights = 1.0 / self.centroid_pos_error_al ** 2

        # sampler interface (matches HGCALogProb convention so things don't break)
        self.hipparcos_epoch = np.array([])   # empty
        self.gaia_epoch = epochs_decyr        # DR4 scan epochs [decimal yr]

        # priors for the four linear astrometric parameters
        self.extra_param_priors=(
            priors.GaussianPrior(catalog_ra_off,catalog_ra_off_err,no_negatives=False),
            priors.GaussianPrior(catalog_dec_off,catalog_dec_off_err,no_negatives=False),
            priors.GaussianPrior(catalog_pmra,catalog_pmra_err,no_negatives=False),
            priors.GaussianPrior(catalog_pmdec,catalog_pmdec_err,no_negatives=False),
        )


    def _save(self, hf):
        """Save to an open HDF5 file."""
        hf.attrs["gaia_num"] = self.gaia_num
        hf.attrs["dr"] = "dr4"
        hf.attrs["dr4_ref_epoch_jd"] = self.ref_epoch_jd

        dr4_dat = read(self.dr4_filepath, converters={"*": [int, float, bytes]})
        hf.create_dataset("Gaia_DR4", data=dr4_dat)

    def compute_lnlike(self, raoff_model, deoff_model, samples, param_idx):
        """
        Compute the Gaussian lnlike of the DR4 along-scan data.
        The four astrometric offsets (``dr4_ra_off``, ``dr4_dec_off``, ``dr4_pmra``, ``dr4_pmdec``)
        are read from the MCMC state vector; their Gaussian priors are evaluated
        separately by the sampler's standard prior machinery.

        Args:
            raoff_model (np.array): NxM primary RA offsets from the
                barycenter due to orbital motion [mas].
            deoff_model (np.array): NxM primary Dec offsets [mas].
            samples (np.array): current parameter vector.
            param_idx (dict): parameter-name to index mapping.

        Returns:
            float: log-likelihood summed over all scans.
        """
        # orbital parameters
        plx = samples[param_idx["plx"]]

        # four explicit astrometric parameters, they come from the catalog
        ra_off = samples[param_idx["dr4_ra_off"]]    # [mas]
        dec_off = samples[param_idx["dr4_dec_off"]]   # [mas]
        pmra = samples[param_idx["dr4_pmra"]]        # [mas/yr]
        pmdec = samples[param_idx["dr4_pmdec"]]       # [mas/yr]

        # project orbital perturbation onto along-scan direction
        orbit_al = - (
            raoff_model[:, 0] * self.sin_scan
            + deoff_model[:, 0] * self.cos_scan
        )

        # full along-scan model: position + PM + parallax + orbit
        model_al = (
            ra_off * self.sin_scan # position offset, projected
            + dec_off * self.cos_scan # position offset, projected
            + pmra * self.pm_ra_basis # proper motion, projected
            + pmdec * self.pm_dec_basis # proper motion, projected
            + plx * self.parallax_factor_al # plx and its factor
            + orbit_al # residual due to planet or companion
        )

        # Gaussian lnlike
        residuals = self.centroid_pos_al - model_al
        chi2 = np.sum(residuals ** 2 * self._weights)

        return float(-0.5 * (chi2 + self._log_norm))