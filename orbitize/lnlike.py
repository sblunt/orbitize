import numpy as np
import pdb

"""
This module contains functions for computing log(likelihood).
"""


def chi2_lnlike(data, errors, corrs, model, jitter, seppa_indices, chi2_type='standard'):
    """Compute Log of the chi2 Likelihood

    Args:
        data (np.array): Nobsx2 array of data, where data[:,0] = sep/RA/RV
            for every epoch, and data[:,1] = corresponding pa/DEC/np.nan.
        errors (np.array): Nobsx2 array of errors for each data point. Same
                format as ``data``.
        corrs (np.array): Nobs array of Pearson correlation coeffs
                between the two quantities. If there is none, can be None.
        model (np.array): Nobsx2xM array of model predictions, where M is the \
                number of orbits being compared against the data. If M is 1, \
            ``model`` can be 2 dimensional.
        jitter (np.array): Nobsx2xM array of jitter values to add to errors.
            Elements of array should be 0 for for all data other than stellar \
            rvs.
        seppa_indices (list): list of epoch numbers whose observations are
            given in sep/PA. This list is located in System.seppa.
        chi2_type (string): the format of chi2 to use is either 'standard' or \
            'log'

    Returns:
        np.array: Nobsx2xM array of chi-squared values.

    .. note::

        (1) **Example**: We have 8 epochs of data for a system. OFTI returns an
        array of 10,000 sets of orbital parameters. The ``model`` input for
        this function should be an array of dimension 8 x 2 x 10,000.

        (2) Chi2_log: redefining chi-sqaured in log scale may give a more stable optimization. \
        This works on separation and position angle data (seppa) not right ascension and declination \
        (radec) data, but it is possible to convert between the two within Orbitize! using the \
        function 'orbitize.system'radec2seppa' (see docuemntation). This implementation defines sep chi-squared \
        in log scale, and defines pa chi-sq using complex phase representation.
            log sep chisq = (log sep - log sep_true)^2 / (sep_sigma / sep_true)^2
            pa chisq = 2 * (1 - cos(pa-pa_true))/pa_sigma^2
i
    """

    if np.ndim(model) == 3:
        # move M dimension to the primary axis, so that numpy knows to iterate over it
        model = np.rollaxis(model, 2, 0)  # now MxNobsx2 in dimensions
        jitter = np.rollaxis(jitter, 2, 0)
        third_dim = True
    elif np.ndim(model) == 2:
        model.shape = (1,) + model.shape
        jitter.shape = (1,) + jitter.shape
        third_dim = False

    if chi2_type == 'standard':
        residual = (data - model)
        # if there are PA values, we should take the difference modulo angle wrapping
        if np.size(seppa_indices) > 0:
            residual[:, seppa_indices, 1] = (residual[:, seppa_indices, 1] + 180.) % 360. - 180.

        sigma2 = errors**2 + jitter**2 # diagonal error term

        if corrs is None:
            # including the second term of chi2
            # the sqrt() in the log() means we don't need to multiply by 0.5
            chi2 = -0.5 * residual**2 / sigma2 - np.log(np.sqrt(2*np.pi*sigma2))
        else:
            has_no_corr = np.isnan(corrs)
            yes_corr = np.where(~has_no_corr)[0]
            no_corr = np.where(has_no_corr)[0]

            chi2 = np.zeros(residual.shape)
            chi2[:,no_corr] = -0.5 * residual[:,no_corr]**2 / sigma2[:,no_corr] - np.log(np.sqrt(2*np.pi*sigma2[:,no_corr]))

            # analytical solution for 2x2 covariance matrix
            # chi2 = -0.5 * (R^T C^-1 R + ln(det_C))
            chi2[:,yes_corr] = _chi2_2x2cov(residual[:,yes_corr], sigma2[:,yes_corr], corrs[yes_corr])

    elif chi2_type == 'log':
        #using the log version of chi squared
        #split the data up into sep, pa, and rv data using seppa_indices and quant
        sep_data = data[seppa_indices, 0]
        sep_model = model[:, seppa_indices, 0]
        sep_error = errors[seppa_indices, 0]
        pa_data = data[seppa_indices, 1]
        pa_model = model[:, seppa_indices, 1]
        pa_error = errors[seppa_indices, 1]*np.pi/180

        #calculating sep chi squared
        sep_chi2_log = (np.log(sep_data)-np.log(sep_model))**2/(sep_error/sep_data)**2

        #calculting pa chi squared Log
        pa_resid = (pa_model-pa_data +180.) % 360. - 180.
        pa_chi2_log = 2*(1-np.cos(pa_resid*np.pi/180))/pa_error**2

        residual = (data - model)
        sigma2 = errors**2 + jitter**2 # diagonal error term

        chi2 = residual**2/sigma2
        chi2[:,seppa_indices,0] = sep_chi2_log
        chi2[:,seppa_indices,1] = pa_chi2_log

        chi2 = -0.5 * chi2 - np.log(np.sqrt(2*np.pi*sigma2))

    if third_dim:
        # move M dimension back to the last axis
        model = np.rollaxis(model, 0, 3)  # now MxNobsx2 in dimensions
        jitter = np.rollaxis(jitter, 0, 3)
        chi2 = np.rollaxis(chi2, 0, 3)  # same with chi2
    else:
        model.shape = model.shape[1:]
        chi2.shape = chi2.shape[1:]
        jitter.shape = jitter.shape[1:]

    return chi2

def _chi2_2x2cov(residual, var, corrs):
    """
    Analytical solution for when quant1/quant2 have a covariance term
    So we don't need to calculate matrix inverses when the jitter varies depending on the model

    Args:
        residual (np.array): MxNobsx2 array of fit residuals,
        var (np.array): MxNobsx2 array of variance for each residual
        corrs (np.array): Nobs array of off axis Pearson corr coeffs
                          between the two quantities.

    Returns:
        np.array: MxNobsx2 array of chi2. Becuase of x/y coariance, it's impossible to
                         spearate the quant1/quant2 chi2. Thus, all the chi2 is in the first term
                         and the second dimension is 0
    """

    det_C = var[:,:,0] * var[:,:,1] * (1 - corrs**2)

    covs = corrs * np.sqrt(var[:,:,0]) * np.sqrt(var[:,:,1])
    chi2 = (residual[:,:,0]**2 * var[:,:,1] + residual[:,:,1]**2 * var[:,:,0] - 2 * residual[:,:,0] * residual[:,:,1] * covs)/det_C

    chi2 += np.log(det_C) + 2 * np.log(2*np.pi) # extra factor of 2 since quant1 and quant2 in each element of chi2.

    chi2 *= -0.5

    chi2 = np.stack([chi2, np.zeros(chi2.shape)], axis=2)

    return chi2

def chi2_norm_term(errors, corrs):
    """
    Return only the normalization term of the Gaussian likelihood: 

    .. math::

        -log(\\sqrt(2\\pi*error^2)) 

    or 

    .. math::
    
        -0.5 * (log(det(C)) + N * log(2\\pi))

    Args:
        errors (np.array): Nobsx2 array of errors for each data point. Same
                format as ``data``.
        corrs (np.array): Nobs array of Pearson correlation coeffs
                between the two quantities. If there is none, can be None.

    Returns:
        float: sum of the normalization terms
    """
    sigma2 = errors**2

    if corrs is None:
        norm = -np.log(np.sqrt(2*np.pi*sigma2))
    else:
        has_no_corr = np.isnan(corrs)
        yes_corr = np.where(~has_no_corr)[0]
        no_corr = np.where(has_no_corr)[0]

        norm = np.zeros(errors.shape)
        norm[no_corr] = -np.log(np.sqrt(2*np.pi*sigma2[no_corr]))

        det_C = sigma2[yes_corr[0], 0] * sigma2[yes_corr[0],1] * (1 - corrs[yes_corr]**2)
        norm[yes_corr,0] = -0.5 * (det_C + 2 * np.log(2 * np.pi)) # extra factor of 2 since quant1 and quant2 in each element of chi2.

    return np.sum(norm)
