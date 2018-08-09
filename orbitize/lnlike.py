import numpy as np

def chi2_lnlike(data, errors, model, seppa_indices):
    """Log of the Chi2 Likelihood Computation

    Args:
        data (np.array): Nobsx2 array of data, where: 
        	data[:,0] = sep/RA/RV for every epoch
        	data[:,1] = corresponding pa/DEC/np.nan
        errors (np.array): Nobsx2 array of errors for each data point. Same
        	format as ``data``
        model (np.array): Nobsx2xM array of model predictions, where M is the
        	number of orbits being compared against the data. If M is 1, ``model``
            can be 2 dimensional.
        seppa_indices (list): list of epoch numbers whose observations are given in 
            sep/PA. This list is located in System.seppa.

    Returns:
    	(np.array): Nobsx2xM array of chi-squared values. 

    Example:
    	We have 8 epochs of data for a system. OFTI returns an array of 
    	10,000 sets of orbital parameters. The ``model`` input for this
    	function should be an array of dimension 8 x 2 x 10,000.

    """
    if np.ndim(model) == 3:
        # move M dimension to the primary axis, so that numpy knows to iterate over it
        model = np.rollaxis(model, 2, 0) # now MxNobsx2 in dimensions

    chi2 = (data - model)**2/errors**2

    # if there are PA values, we should take the difference modulo angle wrapping
    if np.size(seppa_indices) > 0:
        chi2[seppa_indices, 1] = (np.arctan2(
            np.sin(data[seppa_indices, 1] - model[seppa_indices, 1]), 
            np.cos(data[seppa_indices, 1] - model[seppa_indices, 1])
        )**2 / errors[seppa_indices, 1]**2)[:,:,None]

    if np.ndim(model) == 3:
        # move M dimension back to the last axis
        model = np.rollaxis(model, 0, 3) # now MxNobsx2 in dimensions
        chi2 = np.rollaxis(chi2, 0, 3) # same with chi2

    return chi2

