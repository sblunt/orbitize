import numpy as np

"""
This module contains functions for computing log(likelihood).
"""


def chi2_lnlike(data, errors, corrs, model, jitter, seppa_indices):
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

    Returns:
        np.array: Nobsx2xM array of chi-squared values.

    .. note::

        **Example**: We have 8 epochs of data for a system. OFTI returns an
        array of 10,000 sets of orbital parameters. The ``model`` input for
        this function should be an array of dimension 8 x 2 x 10,000.

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
    -log(sqrt(2pi*error^2)) or -0.5 * (log(det(C)) + N * log(2pi))

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