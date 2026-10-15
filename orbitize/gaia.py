import os
import numpy as np
from astropy import units as u
import astropy.io.fits as fits
import astropy.time as time
from astropy.io.ascii import read
from astropy.coordinates import get_body_barycentric_posvel
import numpy.linalg
from orbitize import priors
from orbitize import DATADIR, read_input
import orbitize.lnlike


class GaiaLogProb(object):
    """
    Class to compute the log probability of an orbit with respect to a single
    astrometric position point from Gaia. To be used in conjunction with
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
        gaia_data (dict): a dictionary of Gaia astometry in the
            following form:
                gaia_data = {
                    'ra': 139.4 # RA in degrees
                    'dec': 139.4 # Dec in degrees
                    'ra_error': 0.004 # RA error in mas
                    'dec_error': 0.004 # Dec error in mas
                }

    Written: Sarah Blunt, 2021
    """

    def __init__(self, gaia_num, hiplogprob, gaia_data, dr="dr2"):
        self.gaia_num = gaia_num
        self.hiplogprob = hiplogprob
        self.dr = dr
        self.gaia_data = gaia_data

        if self.dr == "edr3":
            self.gaia_epoch = 2016.0
        elif self.dr == "dr2":
            self.gaia_epoch = 2015.5
        else:
            raise ValueError("`dr` must be either `dr2` or `edr3`")
        self.hipparcos_epoch = 1991.25

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
        gaia_data_group = hf.create_group("gaia_data")
        gaia_data_group.update(self.gaia_data)
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

        return np.sum(lnlike) # TODO: figure out what is adding a dimensions


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
        hgca_filepath (str): path to HGCA catalog FITS file
            (can be downloaded from 
            https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/42/HGCA_vEDR3.fits).
            If None, will check for the file in `orbitize.DATADIR` (default: None)
        hgca_entry (dict): Dictionary of the HGCA catalog entry for this target.
            Can be used to initialize without using the full HGCA catalog FITS file.
            (default: None)

    Written: Jason Wang, 2022
    """

    def __init__(self, hip_id, hiplogprob, gost_filepath, hgca_filepath=None, hgca_entry=None):
        if hgca_entry is None:
            # use default HGCA catalog if not supplied
            if hgca_filepath is None:
                # check orbitize.DATAIDR
                hgca_filepath = os.path.join(DATADIR, "HGCA_vEDR3.fits")
            # grab the entry from the HGCA
            assert os.path.exists(hgca_filepath), "HGCA Catalog Not Found"
            with fits.open(
                hgca_filepath, ignore_missing_simple=True, ignore_missing_end=True
            ) as hdulist:
                hgtable = hdulist[1].data
            entry_raw = hgtable[np.where(hgtable["hip_id"] == hip_id)]
            # check we matched on a single target. mainly check if we typed hip id number incorrectly
            if len(entry_raw) != 1:
                raise ValueError(
                    "HIP {0} encountered {1} matches. Expected 1 match.".format(
                        hip_id, len(entry_raw)
                    )
                )
            cols = entry_raw.columns.names
            entry = dict(zip(cols, entry_raw[0]))
        else:
            if not isinstance(hgca_entry, dict):
                raise TypeError("hgca_entry must be a dict {}".format(hgca_entry))
            entry = hgca_entry
        self.hgca_entry = entry
        self.hip_id = hip_id

        # grab the relevant PM and uncertainties from HGCA
        self.hip_pm = np.array([entry["pmra_hip"], entry["pmdec_hip"]])
        self.hip_pm_err = np.array(
            [entry["pmra_hip_error"], entry["pmdec_hip_error"]]
        )
        hip_radec_cov = (
            entry["pmra_pmdec_hip"]
            * entry["pmra_hip_error"]
            * entry["pmdec_hip_error"]
        )

        self.hg_pm = np.array([entry["pmra_hg"], entry["pmdec_hg"]])
        self.hg_pm_err = np.array(
            [entry["pmra_hg_error"], entry["pmdec_hg_error"]]
        )
        hg_radec_cov = (
            entry["pmra_pmdec_hg"]
            * entry["pmra_hg_error"]
            * entry["pmdec_hg_error"]
        )

        self.gaia_pm = np.array([entry["pmra_gaia"], entry["pmdec_gaia"]])
        self.gaia_pm_err = np.array(
            [entry["pmra_gaia_error"], entry["pmdec_gaia_error"]]
        )
        gaia_radec_cov = (
            entry["pmra_pmdec_gaia"]
            * entry["pmra_gaia_error"]
            * entry["pmdec_gaia_error"]
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
        self.gaia_epoch_ra = entry["epoch_ra_gaia"]
        self.gaia_epoch_dec = entry["epoch_dec_gaia"]
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
        self.hipparcos_epoch_ra = entry["epoch_ra_hip"]
        self.hipparcos_epoch_dec = entry["epoch_dec_hip"]
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

        HGCA_entry_group = hf.create_group("HGCA_entry")
        HGCA_entry_group.update(self.hgca_entry)

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

    @staticmethod
    def _linear_pm_fit(
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

class DR4LogProb(object):
    r"""
    Class to compute the log probability of an orbit with respect to Gaia DR4
    astrometric measurements (epoch astrometry).
    The four linear astrometric parameters (position and proper motion in
    RA* and Dec) are fitted explicitly with uniform priors by default.
    Users can customize these through system.sys_priors before constructing
    the sampler.

    The forward model for each scan at epoch i is:

    .. math::

        \eta_i =
        (\alpha_0 + \mu_{\alpha*}\Delta t_i)\sin\psi_i
        + (\delta_0 + \mu_\delta\Delta t_i)\cos\psi_i
        + \varpi P_i + \eta_{\mathrm{orb},i}

    where:

    - :math:`\eta_i` is the predicted along-scan position [mas].
    - :math:`\alpha_0` and :math:`\delta_0` are the position offsets
      at the reference epoch: ``alpha0`` and ``delta0`` [mas].
    - :math:`\mu_{\alpha*}` and :math:`\mu_\delta` are the proper
      motions: ``pm_ra`` and ``pm_dec`` [mas/yr].
    - :math:`\Delta t_i = (t_i-t_{\mathrm{ref}})/365.25` is the time
      offset [Julian yr], with both epochs in MJD(TCB) and
      ``t_ref = ref_epoch_mjd``.
    - :math:`\psi_i` is the scan position angle [rad].
    - :math:`\varpi` is ``plx`` [mas], and :math:`P_i` is the
      dimensionless along-scan parallax factor.
    - :math:`\eta_{\mathrm{orb},i}` is the orbital contribution [mas],
      computed as ``-raoff_model[i, 0]*sin(psi_i)
      - deoff_model[i, 0]*cos(psi_i)`` using the likelihood's
      existing sign convention.

    These parameters are added by the fitting basis when System is
    constructed with gaia=dr4. Users can customize their priors through
    system.sys_priors before constructing the sampler.

    Pass this object into the gaia keyword of orbitize.system.System.
    You must 'set fit_secondary_mass=True' so that the star's barycentric
    wobble is computed.

    Args:
        gaia_num (int): Gaia source ID
        dr4_filepath (str or astropy.table.Table): Raw OHP CSV (JD timestamps),
            or a table returned by ``from_ohp_file`` / ``from_dl2``
            (MJD timestamps). Times are normalized once to MJD(TCB).
            Required columns: 'obs_time_tcb', 'centroid_pos_al',
            'centroid_pos_error_al', 'parallax_factor_al', 'scan_pos_angle',
            'field_of_view'. MJD tables must carry ``meta['time_format']='mjd'``.
        ref_epoch_jd (float, optional): Legacy JD(TCB) reference epoch;
            converted to MJD on input. Prefer ``ref_epoch_mjd`` for new code.
        ref_epoch_mjd (float, optional): Reference epoch in MJD(TCB) for the
            linear astrometric model. Default 57936.375 (J2017.5).
        jitter (float): fixed along-scan jitter [mas] added in quadrature to
            ``centroid_pos_error_al``. Default 0, preserving the original
            likelihood.
        jitter_prior (orbitize.priors.Prior, optional): prior for a fitted
            along-scan jitter parameter [mas]. When supplied, ``dr4_jitter``
            is registered as an additional System parameter. ``jitter`` must
            remain zero when fitting the jitter.

    Other data and the orbit's ``tau_ref_epoch`` must use a compatible time
    scale for joint fits. This class does not convert RV or relative-astrometry
    timestamps whose time scale is unspecified.

    Written: Clarissa Do O, 2026
    """

    def __init__(
        self,
        gaia_num,
        dr4_filepath,
        ref_epoch_jd=None,
        jitter=0.0,
        jitter_prior=None,
        ref_epoch_mjd=None,
    ):
        if ref_epoch_jd is not None:
            if ref_epoch_mjd is not None:
                raise ValueError("Set only one of ref_epoch_mjd and ref_epoch_jd.")
            ref_epoch_mjd = ref_epoch_jd - 2400000.5
        if ref_epoch_mjd is None:
            ref_epoch_mjd = 57936.375  # J2017.5 in MJD(TCB)
        if not np.isfinite(jitter) or jitter < 0:
            raise ValueError("jitter must be a finite, non-negative value in mas")
        if jitter_prior is not None and jitter != 0:
            raise ValueError("set either a fixed jitter or jitter_prior, not both")
        if jitter_prior is not None and not isinstance(jitter_prior, priors.Prior):
            raise TypeError("jitter_prior must be an orbitize.priors.Prior")

        self.gaia_num = gaia_num
        self.dr4_filepath = dr4_filepath
        self.ref_epoch_mjd = ref_epoch_mjd
        self.time_scale = "tcb"
        self.jitter = float(jitter)
        self.fit_jitter = jitter_prior is not None
        self.jitter_prior = jitter_prior
        # Normalize raw OHP files or already-read Gaia tables once to MJD(TCB).
        self.data_table = read_input.from_ohp_file(dr4_filepath)
        dr4_dat = self.data_table
        self.epochs_mjd = np.array(dr4_dat["obs_time_tcb"])
        self.n_obs = len(self.epochs_mjd)

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
        self.dt_yr = (self.epochs_mjd - self.ref_epoch_mjd) / 365.25

        # pre-compute scan projection of PM basis vectors
        self.pm_ra_basis = self.dt_yr * self.sin_scan  # pmra, projected
        self.pm_dec_basis = self.dt_yr * self.cos_scan  # pmdec, projected

        # Decimal years are retained for display/legacy callers, not orbit times.
        self.hipparcos_epoch = np.array([])  # empty
        self.gaia_epoch = time.Time(
            self.epochs_mjd, format="mjd", scale=self.time_scale
        ).decimalyear

    def _save(self, hf):
        """Save to an open HDF5 file."""
        hf.attrs["gaia_num"] = self.gaia_num
        hf.attrs["dr"] = "dr4"
        hf.attrs["dr4_ref_epoch_mjd"] = self.ref_epoch_mjd
        hf.attrs["dr4_time_format"] = "mjd"
        hf.attrs["dr4_time_scale"] = self.time_scale
        hf.attrs["dr4_jitter"] = self.jitter
        hf.attrs["dr4_fit_jitter"] = self.fit_jitter

        dr4_dat = self.data_table.copy()
        dr4_dat.convert_unicode_to_bytestring()
        hf.create_dataset("Gaia_DR4", data=dr4_dat.as_array())

    def compute_lnlike(self, raoff_model, deoff_model, samples, param_idx):
        """
        Compute the Gaussian log-likelihood of the DR4 along-scan data.
        The position offsets (``alpha0``, ``delta0``), proper motions
        (``pm_ra``, ``pm_dec``), and parallax (``plx``) are read from
        the parameter vector. Their priors are evaluated separately
        by the sampler. If ``jitter_prior`` was supplied at construction, ``dr4_jitter``
        is read from the parameter vector. Otherwise, the fixed jitter
        is used. Jitter is added in quadrature to each AL uncertainty.

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

        # sampled astrometric parameters
        ra_off = samples[param_idx["alpha0"]]  # mas
        dec_off = samples[param_idx["delta0"]]  # mas
        pmra = samples[param_idx["pm_ra"]]  # mas/yr
        pmdec = samples[param_idx["pm_dec"]]  # mas/yr
        # project orbital perturbation onto along-scan direction
        orbit_al = -(
            raoff_model[:, 0] * self.sin_scan
            + deoff_model[:, 0] * self.cos_scan
        )

        # full along-scan model: position + PM + parallax + orbit
        model_al = (
            ra_off * self.sin_scan  # position offset, projected
            + dec_off * self.cos_scan  # position offset, projected
            + pmra * self.pm_ra_basis  # proper motion, projected
            + pmdec * self.pm_dec_basis  # proper motion, projected
            + plx * self.parallax_factor_al  # plx and its factor
            + orbit_al  # residual due to planet or companion
        )

        # Gaussian lnlike. The normalization must be recomputed when jitter is
        # fitted because the effective uncertainties then vary between samples.
        residuals = self.centroid_pos_al - model_al
        if self.fit_jitter:
            jitter = samples[param_idx["dr4_jitter"]]
        else:
            jitter = self.jitter

        variance = self.centroid_pos_error_al**2 + jitter**2
        chi2 = np.sum(residuals**2 / variance)
        log_norm = np.sum(np.log(2.0 * np.pi * variance))

        return float(-0.5 * (chi2 + log_norm))
