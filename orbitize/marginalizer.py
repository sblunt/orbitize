import abc
import numpy as np
from astropy import units
import orbitize.basis as basis
import orbitize.kepler as kepler
import warnings

"""
This module defines marginalizations to draw
samples of non-marginalized parameters and compute appropriate probability
"""


class Marginalizer(abc.ABC):
    """
    Abstract Base Class for marginalizer objects.
    All marginalizer objects should inherit from this class.

    Written: Eshel Dror, 2026
    """

    QUANT_TYPES = []
    REQ_PARAMS = []
    MARG_PARAMS = []
    LNLIKE_TYPE = None
    MULTIBODY = False
    GAIA = False

    @abc.abstractmethod
    def apply(self, param_arr, calculated_values={}):
        """
        Apply the marginalization and get the calculated probabilities and sampled parameters.

        Args:
            param_arr (np.array of float): RxM array of fitting parameters. Where
                R is the number of required parameters in the order of `self.REQ_PARAMS` and
                M is the number of orbits. If M=1 this can be a 1d array.
            calculated_values (dict): Dictionary of calculated values passed through consecutive marginalizers.

        Returns:
            float or np.array of float: log-probability of the likelihood of the parameters
                in `param_arr` and the priors of the marginalized parameters.
            float or np.array of float: log-likelihood of the parameters in `param_arr`
                and the sampled marginalized parameters.
            np.array of float: samples of shape PxM where P is the number of margainalized paramters
                and M is the number of orbits. In the order of `self.MARG_PARAMS`. May be 1D if M=1.
            dict: `calculated_values` including any updates.
        """
        pass

    @staticmethod
    def lnlike(y, X, beta_hat, W, W_inv=None, W_absdet=None, add_error_norm=True):
        """
        Compute log likelihood, excluding volume in marginalization integral (first two terms in \ref{eq:marg_like} in
        PASP review by Blunt 2026):

            -1/2 log|W| - 1/2 chi2_MAP

        Args:
            y: 2*n_obs array of observations (sep on top, PA-g on bottom)
            X: 2*n_obs x 2 design matrix
            beta_hat: length 2 array of conditional MAP values of sma and pan (respectively)
            W: 2n x 2n covariance matrix corresponding to observations y
            W_inv: 2n x 2n inverse of the covariance matrix W, if calculated previously. (default: None)
            add_error_norm: if True, include the -1/2 log|W| term (needed if fitting for jitter
                or want a result consistent with other methods). (default: True).

        Returns:
            float: loglikelihood
        """
        if W_inv is None:
            W_inv = np.linalg.inv(W)
        chi2_map = (y - X @ beta_hat).T @ W_inv @ (y - X @ beta_hat)

        # NOTE: this term can be ignored unless we're fitting
        # for jitter or want a result consistent with other methods
        if add_error_norm:
            if W_absdet is None:
                error_norm_term = -0.5 * np.linalg.slogdet(W)[1]
            else:
                error_norm_term = -0.5 * np.log(W_absdet)
        else:
            error_norm_term = 0

        return -0.5 * chi2_map + error_norm_term

class OFTeI_sma_pan(Marginalizer):
    """
    OFTeI marginalizer for relative astrometry.

    Currently only applicable to 1 non-massive companion.
    REQ_PARAMS = ["ecc1", "inc1", "aop1", "tau1", "plx", "per1"]
    MARG_PARAMS = ["sma1", "pan1"]

    Args:
        epochs (np.array of float): epochs of seppa data
        seps (np.array of float): Separation data
        pas (np.array of float): Position Angle data
        sigma_seps (np.array of float): Standard Deviation of the Separation data
        sigma_pas (np.array of float): Standard Deviation of the Position Angle data
        covar_seps_pas (np.array of float): Covariance of Separation and Position Angle data at the same epoch
    """

    REQ_PARAMS = ["ecc1", "inc1", "aop1", "tau1", "plx", "per1"]
    MARG_PARAMS = ["sma1", "pan1"]
    QUANT_TYPES = ["seppa"]
    LNLIKE_TYPE = "chi2"

    def __init__(
        self,
        epochs,
        seps,
        pas,
        sigma_seps,
        sigma_pas,
        covar_seps_pas,
    ):
        self.epochs = epochs
        self.seps = seps
        self.pas = pas
        self.sigma_seps = sigma_seps
        self.sigma_pas = sigma_pas
        self.covar_seps_pas = covar_seps_pas
        self.n_obs = len(epochs)
        # compute variance-covariance matrix W of observational uncertainties
        self.W = (
            np.diag(np.hstack((self.sigma_seps**2, self.sigma_pas**2)))
            + np.diag(self.covar_seps_pas, k=self.n_obs)
            + np.diag(self.covar_seps_pas, k=-self.n_obs)
        )
        self.W_inv = np.linalg.inv(self.W)

    @staticmethod
    def calc_f_and_g(manoms, ecc, inc, aop, plx):
        """
        Args:
            manoms (np.array of float): array of mean anomalies [rad, 0-2pi]
            ecc (float): eccentricity
            inc (float): inclination [rad, 0-pi]
            aop (float): argument of periastron [rad, 0-2pi]
            plx (float): parallax [mas]

        Return:
            tuple of:
                np.array of float: f (defined in \ref{eq:f_and_g} in PASP review by Blunt 2026)
                np.array of float: g (defined in \ref{eq:f_and_g} in PASP review by Blunt 2026)

        """

        n_dates = np.size(manoms)

        # compute eccentric anomalies (size: n_dates)
        eanom = kepler.calc_ecc_anom(manoms, ecc)

        tanom = 2.0 * np.arctan(
            np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(0.5 * eanom)
        )

        radius = (1.0 - ecc * np.cos(eanom)) * plx

        f = (
            0.5
            * radius
            * np.sqrt(
                3 + np.cos(2 * inc) + 2 * np.cos(2 * (aop + tanom)) * (np.sin(inc)) ** 2
            )
        )

        aoptanom_02pi = ((aop + tanom) % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)

        quadrant_aoptanom = basis.return_quadrant(aoptanom_02pi)
        quadrant_corr = np.zeros(n_dates) - np.pi / 2
        quadrant_corr[quadrant_aoptanom == 1] -= np.pi
        quadrant_corr[quadrant_aoptanom == 4] -= np.pi

        if np.cos(inc) < 0:
            quadrant_corr += np.pi

        g_radians = (
            np.arctan2(
                -1 / np.cos(inc),
                np.tan(aoptanom_02pi),
            )
        ) + quadrant_corr

        g = np.degrees(g_radians) % 360

        return f, g

    @staticmethod
    def calc_cov_matrix_det(n_obs, Y, D, W, W_inv=None):
        """
        Calculate the MAP values of sma and pan and the determinant of the
        covariance matrix describing their conditional posterior uncertainties

        Args:
            n_obs (float): number of observations
            Y (np.array of float): (2 * n_obs) array of observations (sep on top, PA-g on bottom)
            D (np.array of float): (2 * n_obs) x 2 design matrix
            W (np.array of float): 2n x 2n covariance matrix corresponding to observations Y
            W_inv (np.array of float): inverse of `W` if calculated previously (default: None)

        Returns:
            float: MAP Semimajor Axis
            float: MAP Position-Angle of Nodes (Degrees)
            float: log(abs(determinant of cov matrix))
            float: Sample of Semimajor Axis
            float: Sample of Position Angle of Nodes
            float: log(p(samples|data,params))
        """

        map_fit, cov_matrix, precision_matrix = solve_linear_eq(Y, D, W, W_inv)
        sma_map = map_fit[0]
        pan_map = map_fit[1]

        # sample linear parameters
        sma_sample, pan_sample = np.random.multivariate_normal(
            [sma_map, pan_map], cov_matrix
        )

        _, logabsdet_covmatrix = np.linalg.slogdet(cov_matrix)
        logabsdet_covmatrix += 2 * np.log(2 * np.pi)

        res = np.array([sma_sample - sma_map, basis.angle_diff(pan_sample, pan_map)])

        log_conditional_prob = (
            -0.5 * res.T @ precision_matrix @ res - 0.5 * logabsdet_covmatrix
        )

        log_conditional_prob -= (n_obs) * np.log(2 * np.pi)

        return (
            sma_map,
            pan_map,
            logabsdet_covmatrix,
            sma_sample,
            pan_sample,
            log_conditional_prob,
        )

    def apply(self, param_arr, calculated_values):
        shape = param_arr.shape
        if len(shape) > 1:
            raise IndexError("Only support 1 orbit at a time")
        ecc, inc, aop, tau, plx, period_yr = param_arr
        if not period_yr > 0:
            raise ValueError(f"Marginalization encountered a non-positive period {period_yr} in {param_arr=}")
        frac_date = self.epochs / (period_yr * units.yr.to(units.day))
        frac_date %= 1
        manoms = (frac_date - tau) * 2 * np.pi
        manoms %= 2 * np.pi
        f_samples, g_samples = self.calc_f_and_g(
            manoms,
            ecc,
            inc,
            aop,
            plx,
        )
        W = self.W
        W_inv = self.W_inv
        Y = np.hstack((self.seps, basis.angle_diff(self.pas, g_samples)))
        # set up design matrix
        D = np.zeros((2 * self.n_obs, 2))
        D[: self.n_obs, 0] = f_samples
        D[self.n_obs :, 1] = 1

        try:
            (
                smas_map,
                pans_map_degrees,
                logdet_covmatrix,
                sma_sample,
                pan_sample_deg,
                log_conditional_prob,
            ) = self.calc_cov_matrix_det(self.n_obs, Y, D, W, W_inv)
        except np.linalg.LinAlgError as e:
            warnings.warn(
                f"Encountered LinAlgError in marginalizatoin {e.args[0]}, assuming prob=0"
            )
            return -np.inf, -np.inf, None, dict()

        beta_hat = np.hstack((smas_map, pans_map_degrees))

        # compute marginalized likelihood (e.g. Price-Whelan 2017 eq 11)
        chi2lnlike = self.lnlike(Y, D, beta_hat, W, W_inv, add_error_norm=True)

        logprob = (
            chi2lnlike + 0.5 * logdet_covmatrix
        )  # NOTE: the sign here differs from APW+ 2017 (which I think is correct; I think they have a bug)

        pan_sample = np.deg2rad(pan_sample_deg) % (2 * np.pi)
        loglike = logprob + log_conditional_prob

        samples = np.array([sma_sample, pan_sample])

        if not np.isfinite(logprob):
            return -np.inf, -np.inf, samples, calculated_values

        return logprob, loglike, samples, calculated_values


class OFTeI_DR4_gaia(Marginalizer):
    """
    OFTeI marginalizer for Gaia DR4 absolute astrometry.

    Currently only applicable to 1 massive companion.
    REQ_PARAMS = ["ecc1", "inc1", "aop1", "tau1", "per1", "mfrac1"] + ["dr4_jitter"] if fitting jitter
    MARG_PARAMS = ["alpha0", "pm_ra", "delta0", "pm_dec", "plx", "u", "v"]

    where:
    u = plx * sma_photo * cos(pan)
    v = plx * sma_photo * sin(pan)

    Args:
        gaia (orbitize.gaia.DR4LogProb): Gaia DR4 object
    """

    REQ_PARAMS = ["ecc1", "inc1", "aop1", "tau1", "per1", "mfrac1"]
    MARG_PARAMS = ["alpha0", "pm_ra", "delta0", "pm_dec", "plx", "u", "v"]
    LNLIKE_TYPE = "chi2"
    QUANT_TYPES = []
    GAIA = True

    def __init__(
        self,
        gaia,
    ):
        self.gaia = gaia
        if self.gaia.fit_jitter:
            self.REQ_PARAMS.append("dr4_jitter")

    def calc_As_and_Bs(self, manoms, ecc, inc, aop):
        """
        Args:
            manoms (np.array of float): array of mean anomalies [rad, 0-2pi]
            ecc (float): eccentricity
            inc (float): inclination [rad, 0-pi]
            aop (float): argument of periastron [rad, 0-2pi]

        Return:
            tuple of:
                np.array of float: A (defined in \ref{eq:gaia_matrix} in PASP review by Blunt 2026)
                np.array of float: B (defined in \ref{eq:gaia_matrix} in PASP review by Blunt 2026)

        """

        n_dates = np.size(manoms)

        # compute eccentric anomalies (size: n_dates)
        eanom = kepler.calc_ecc_anom(manoms, ecc)

        tanom = 2.0 * np.arctan(
            np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(0.5 * eanom)
        )

        radius = 1.0 - ecc * np.cos(eanom)

        aop_star = np.pi + aop

        As = - radius * ( # TODO: verify negative
            np.cos(tanom + aop_star) * self.gaia.cos_scan
            + np.cos(inc) * np.sin(tanom + aop_star) * self.gaia.sin_scan
        )
        Bs = - radius * (
            np.cos(tanom + aop_star) * self.gaia.sin_scan
            - np.cos(inc) * np.sin(tanom + aop_star) * self.gaia.cos_scan
        )

        return As, Bs

    def calc_cov_matrix_det(self, Y, D, W, W_inv=None):
        """
        Calculate the MAP values of the marginalized parameters and the determinant of the
        covariance matrix describing their conditional posterior uncertainties

        Args:
            Y (np.array of float): (2 * n_obs) array of observations (sep on top, PA-g on bottom)
            D (np.array of float): (2 * n_obs) x 2 design matrix
            W (np.array of float): 2n x 2n covariance matrix corresponding to observations Y
            W_inv (np.array of float): inverse of `W` if calculated previously (default: None)

        Returns:
            np.array of float: MAP of Marginalized Parameters
            float: log(abs(determinant of cov matrix))
            np.array of float: Samples of the Marginalized Parameters
            float: log(p(samples|data,params))
        """
        n_obs = self.gaia.n_obs

        map_fit, cov_matrix, precision_matrix = solve_linear_eq(Y, D, W, W_inv)

        # Sample linear params
        sample = np.random.multivariate_normal(map_fit, cov_matrix)

        _, logabsdet_covmatrix = np.linalg.slogdet(cov_matrix)
        logabsdet_covmatrix += 2 * np.log(2 * np.pi)

        res = sample - map_fit

        log_conditional_prob = (
            -0.5 * res.T @ precision_matrix @ res - 0.5 * logabsdet_covmatrix
        )

        log_conditional_prob -= (n_obs / 2) * np.log(
            2 * np.pi
        )  # Figure out if half n_obs is right

        return map_fit, logabsdet_covmatrix, sample, log_conditional_prob

    def apply(self, param_arr, calculated_values):
        shape = param_arr.shape
        if len(shape) > 1:
            raise IndexError("Only support 1 orbit at a time")
        if self.gaia.fit_jitter:
            ecc, inc, aop, tau, period_yr, mfrac1, jitter = param_arr
        else:
            ecc, inc, aop, tau, period_yr, mfrac1 = param_arr
            jitter = self.gaia.jitter
        if not period_yr > 0:
            raise ValueError(f"Marginalization encountered a non-positive period {period_yr} in {param_arr=}")
        frac_date = self.gaia.epochs_mjd / (period_yr * units.yr.to(units.day))
        frac_date %= 1
        manoms = (frac_date - tau) * 2 * np.pi
        manoms %= 2 * np.pi
        As, Bs = self.calc_As_and_Bs(
            manoms,
            ecc,
            inc,
            aop,
        )

        variances = self.gaia.centroid_pos_error_al**2 + jitter**2
        W = np.diag(variances)
        W_inv = np.diag(1 / variances)
        W_det = np.prod(variances)

        n_obs = self.gaia.n_obs

        D = np.zeros((n_obs, 7))
        D[:, 0] = self.gaia.sin_scan
        D[:, 1] = self.gaia.pm_ra_basis
        D[:, 2] = self.gaia.cos_scan
        D[:, 3] = self.gaia.pm_dec_basis
        D[:, 4] = self.gaia.parallax_factor_al
        D[:, 5] = As
        D[:, 6] = Bs

        Y = self.gaia.centroid_pos_al

        try:
            map_fit, logabsdet_covmatrix, samples, log_conditional_prob = (
                self.calc_cov_matrix_det(Y, D, W, W_inv)
            )
        except np.linalg.LinAlgError as e:
            warnings.warn(
                f"Encountered LinAlgError in marginalizatoin {e.args[0]}, assuming prob=0"
            )
            return -np.inf, -np.inf, None, {}

        chi2lnlike = self.lnlike(Y, D, map_fit, W, W_inv, W_det, add_error_norm=True)

        logprob = chi2lnlike + 0.5 * logabsdet_covmatrix

        loglike = logprob + log_conditional_prob

        if not np.isfinite(logprob):
            return -np.inf, -np.inf, samples, calculated_values

        return logprob, loglike, samples, calculated_values


def solve_linear_eq(Y, A, C, C_inv=None):
    """
    Solve for the MAP of beta with
    Y - A @ beta ~ Norm(0,C)

    Args:
        Y: Data
        A: Design Matrix
        C: Covariance Matrix of the Data
        C_inv: Inverse of C if calculated previously (default: None)

    Returns:
        np.array: MAP beta
        np.array: covariance matrix of p(beta|Y,A)
        np.array: precision matrix (inverse of covariance matrix)
    """
    if C_inv is None:
        C_inv = np.linalg.inv(C)
    prec_matrix = A.T @ C_inv @ A
    cov_matrix = np.linalg.inv(prec_matrix)  # variance on diag, covar on off-diag
    best_fit = cov_matrix @ (A.T @ C_inv @ Y)
    return best_fit, cov_matrix, prec_matrix


def compatible_params(marginalizers, input_params, data_quants):
    # TODO: Multibody, quant_type
    params = set(input_params)
    quants = set(data_quants)
    for marginalizer in marginalizers:
        req_params = set(marginalizer.REQ_PARAMS)
        marg_params = set(marginalizer.MARG_PARAMS)
        marg_quants = set(marginalizer.QUANT_TYPES)
        if (
            not (req_params - params)
            and not (marg_params & params)
            and (not (marg_quants) or (quants & marg_quants))
        ):
            # TODO: differentiate between requiring all quant types vs any quant types
            params |= set(marginalizer.MARG_PARAMS)
        else:
            return False
    return True
