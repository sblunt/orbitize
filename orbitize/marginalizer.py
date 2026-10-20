import abc
import numpy as np
from astropy import units as u
import orbitize.basis as basis
import orbitize.kepler as kepler

"""
This module defines marginalizations to draw
samples of non-marginalized parameters and compute appropriate probability
"""

class Marginalizer(abc.ABC):

    QUANT_TYPES = []
    JITTER_PARAMS = []
    REQ_PARAMS = []
    MARG_PARAMS = []
    LNLIKE_TYPE = None
    MULTIBODY = False
    
    @abc.abstractmethod
    def apply(self, param_arr, jitter_arr=None, calculated_values={}):
        pass

class Marg1(Marginalizer):

    REQ_PARAMS = ["tau", "per"]
    MARG_PARAMS = ["sma", "pan"]

class Marg2(Marginalizer):

    REQ_PARAMS = ["tau", "sma", "pan"]
    MARG_PARAMS = ["gamma"]

class MargConstNoInput(Marginalizer):

    MARG_PARAMS = ["test"]

    def apply(self, param_arr, jitter_arr, calculated_values):
        shape = param_arr.shape
        if len(shape) > 1:
            n_orbits = shape[1]
            return 0, np.zeros((n_orbits)), np.ones((n_orbits)), calculated_values
        else:
            return 0, 0, 1, calculated_values
        
class MargPriorNoInput(Marginalizer):

    MARG_PARAMS = ["test"]

    def __init__(self, priors):
        self.dim = len(priors)
        self.MARG_PARAMS = ["test{}".format(i) for i in range(self.dim)]
        self.priors = priors

    def apply(self, param_arr, jitter_arr, calculated_values):
        shape = param_arr.shape
        if len(shape) > 1:
            n_orbits = shape[1]
            assert n_orbits > 1
            samples = np.array([[prior.draw_samples(1).item() for i in range(n_orbits)] for prior in self.priors])  
            assert samples.shape == (self.dim, n_orbits)  
            assert not np.isnan(samples).any(), samples
            return 0, np.zeros((n_orbits)), samples, calculated_values
        else:
            samples = np.array([prior.draw_samples(1).item() for prior in self.priors])
            assert samples.shape == (self.dim,), samples
            assert not np.isnan(samples).any(), samples
            return 0, 0, samples, calculated_values

class OFTeI_sma_pan(Marginalizer):

    REQ_PARAMS = ["ecc1", "inc1", "aop1", "tau1", "plx", "per1"]
    MARG_PARAMS = ["sma1", "pan1"]
    QUANT_TYPES = ["seppa"]
    LNLIKE_TYPE = "chi2"

    def __init__(self, epochs, seps, pas, sigma_seps, sigma_pas, covar_seps_pas, sma_prior=None, pan_prior=None, mass_prior=None, implicit_mass_prior=None):
        self.sma_prior = sma_prior
        self.pan_prior = pan_prior
        self.mass_prior = mass_prior
        self.implicit_mass_prior = implicit_mass_prior
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
                float: f (defined in \ref{eq:f_and_g} in PASP review)
                float: g (defined in \ref{eq:f_and_g} in PASP review)

        """

        n_dates = np.size(manoms)

        # compute eccentric anomalies (size: n_dates)
        eanom = kepler.calc_ecc_anom(manoms, ecc, use_c=False)

        tanom = 2.0 * np.arctan(np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(0.5 * eanom))

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
    def calc_cov_matrix_det(seps, pas, fs, gs, W, W_inv=None):
        """
        Calculate the MAP values of sma and pan and the determinant of the
        covariance matrix describing their conditional posterior uncertainties

        Args:
            seps (np.array of float): separation measurements [mas]
            pas (np.array of float): position angle measurements [deg]
            fs (np.array of float): (defined in \ref{eq:f_and_g} in PASP review), values at observation epochs
            gs (np.array of float): (defined in \ref{eq:f_and_g} in PASP review), values at observation epochs
            W: 2n x 2n covariance matrix corresponding to observations y

        Returns:
            float: log(abs(determinant of cov matrix))
        """

        n_obs = len(seps)

        # set up design matrix
        D = np.zeros((2 * n_obs, 2))
        D[:n_obs, 0] = fs
        D[n_obs:, 1] = 1

        Y = np.hstack((seps, basis.angle_diff(pas, gs)))

        def solve_linear_eq(Y, A, C, C_inv=None):
            if C_inv is None:
                C_inv = np.linalg.inv(C)
            cov_matrix = np.linalg.inv(
                A.T @ C_inv @ A
            )  # variance on diag, covar on off-diag
            best_fit = cov_matrix @ (A.T @ C_inv @ Y)
            return best_fit, cov_matrix

        map_fit, cov_matrix = solve_linear_eq(Y, D, W, W_inv)
        sma_map = map_fit[0]
        pan_map = map_fit[1]

        # sample linear parameters
        sma_sample, pan_sample = np.random.multivariate_normal(
            [sma_map, pan_map], cov_matrix
        )

        _, logabsdet_covmatrix = np.linalg.slogdet(
            cov_matrix
        )  # NOTE: the 2pi factor in the APW paper is constant; doesn't affect overall lnlike

        return sma_map, pan_map, logabsdet_covmatrix, sma_sample, pan_sample

    @staticmethod
    def lnlike(y, X, beta_hat, W, W_inv=None, add_error_norm=False):
        """
        Compute log likelihood, excluding volume in marginalization integral (first two terms in \ref{eq:marg_like} in
        PASP review):

            -1/2 log|W| - 1/2 chi2_MAP

        Args:
            y: 2*n_obs array of observations (sep on top, PA-g on bottom)
            X: 2*n_obs x 2 design matrix
            beta_hat: length 2 array of conditional MAP values of sma and pan (respectively)
            W: 2n x 2n covariance matrix corresponding to observations y
            add_error_term: if True, include the -1/2 log|W| term (needed if fitting for jitter). Default False.as_integer_ratio

        Returns:
            float: loglikelihood
        """
        if W_inv is None:
            W_inv = np.linalg.inv(W)
        chi2_map = (y - X @ beta_hat).T @ W_inv @ (y - X @ beta_hat)

        # NOTE: this term can be ignored unless we're fitting for jitter
        if add_error_norm:
            error_norm_term = -0.5 * np.log(np.det(W))
        else:
            error_norm_term = 0

        return -0.5 * chi2_map + error_norm_term

    def apply(self, param_arr, jitter_arr, calculated_values):
        assert jitter_arr.shape[0] == 0, "No jitter supported"
        shape = param_arr.shape
        if len(shape) > 1:
            raise IndexError("Only support 1 orbit at a time")
        ecc, inc, aop, tau, plx, period_yr = param_arr
        assert period_yr > 0
        frac_date = self.epochs / (period_yr * u.yr.to(u.day))
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

        try:
            smas_map, pans_map_degrees, logdet_covmatrix, sma_sample, pan_sample_deg = (
                self.calc_cov_matrix_det(self.seps, self.pas, f_samples, g_samples, W, W_inv)
            )
        except np.linalg.LinAlgError:
            return -np.inf, None, None

        # set up design matrix
        X = np.zeros((2 * self.n_obs, 2))
        X[:self.n_obs, 0] = f_samples
        X[self.n_obs:, 1] = 1

        y = np.hstack((self.seps, basis.angle_diff(self.pas, g_samples)))
        beta_hat = np.hstack((smas_map, pans_map_degrees))

        # compute marginalized likelihood (e.g. Price-Whelan 2017 eq 11)
        chi2lnlike = self.lnlike(y, X, beta_hat, W, W_inv)

        logprob = (
            chi2lnlike + 0.5 * logdet_covmatrix
        )  # NOTE: the sign here differs from APW+ 2017 (which I think is correct; I think they have a bug)

        pan_sample = np.deg2rad(pan_sample_deg) % (2 * np.pi)

        # apply sma/pan priors
        if self.sma_prior is not None:
            logprob += np.sum(self.sma_prior.compute_lnprob(np.array([sma_sample])))
        if self.pan_prior is not None:
            logprob += np.sum(self.pan_prior.compute_lnprob(np.array([pan_sample])))

        # apply total mass prior
        mtot_sample = sma_sample**3 / (period_yr) ** 2
        if self.mass_prior is not None:
            logprob -= np.sum(self.implicit_mass_prior.compute_lnprob(
                np.array([mtot_sample])
            ))  # "undo" the effective prior
            logprob += np.sum(self.mass_prior.compute_lnprob(
                np.array([mtot_sample])
            ))  # apply the user-defined prior

        loglike = logprob # TODO
        samples = np.array([sma_sample, pan_sample])

        if not np.isfinite(logprob):
            return -np.inf, -np.inf, samples, calculated_values

        return logprob, loglike, samples, calculated_values


def compatible_params(marginalizers, input_params, data_quants):
    # TODO: Multibody, quant_type
    params = set(input_params)
    quants = set(data_quants)
    for marginalizer in marginalizers:
        req_params = set(marginalizer.REQ_PARAMS)
        marg_params = set(marginalizer.MARG_PARAMS)
        marg_quants = set(marginalizer.QUANT_TYPES)
        if len(req_params - params) == 0 and len(marg_params & params) == 0 and (len(marg_quants) == 0 or len(quants & marg_quants) > 0):
            # TODO: differentiate between requiring all quant types vs any quant types
            params |= set(marginalizer.MARG_PARAMS)
        else:
            return False
    return True