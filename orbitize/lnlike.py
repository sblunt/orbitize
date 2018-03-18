def chi2_lnlike(data, errors, model):
    """Log of the Chi2 Likelihood Computation

    Args:
        data (np.array): Nobsx2 array of data, where: 
        	data[:,0] = sep/RA/RV for every epoch
        	data[:,1] = corresponding pa/DEC/np.nan
        errors (np.array): Nobsx2 array of errors for each data point. Same
        	format as ``data``
        model (np.array): MxNobsx2 array of model predictions, where M is the
        	number of orbits being compared against the data. M can be 1.

    Returns:
    	(np.array): MxNobsx2 array of chi-squared values. 

    Example:
    	We have 8 epochs of data for a system. OFTI returns an array of 
    	10,000 sets of orbital parameters. The ``model`` input for this
    	function should be an array of dimension 10,000 x 8 x 2.

    """

    chi2 = (data - model)**2/errors**2

    return chi2

