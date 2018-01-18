def chi2_likelihood(data, errors, model):
    """Log of the Chi2 Likelihood Computation

    Args:
        data (np.array): 1-D array of data
        errors (np.array): 1-D array of errors for each data point
        model (np.array): 1-D array of model predictions
    """

    chi2 = (data - model)**2/errors**2

    return chi2

