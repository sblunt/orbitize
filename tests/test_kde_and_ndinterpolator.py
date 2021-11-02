import numpy as np
import pytest
from scipy.stats import norm as nm
from scipy.stats import gaussian_kde
from scipy.interpolate import NearestNDInterpolator

import orbitize.priors as priors
import orbitize.results as results
import orbitize

import pandas as pd



def test_kde():
    # Read RV posteriors
    pdf_fromRadVel = pd.read_csv(orbitize.DATADIR+'sample_radvel_chains.csv.bz2', compression='bz2', index_col=0)
    per1 = pdf_fromRadVel.per1
    k1 = pdf_fromRadVel.k1 # Doppler semi-amplitude
    secosw1 = pdf_fromRadVel.secosw1
    sesinw1 = pdf_fromRadVel.sesinw1
    tc1 = pdf_fromRadVel.tc1
    
    # Put together posteriors to initialize KDE
    len_pdf = len(pdf_fromRadVel)
    total_params = 5
    values = np.empty((total_params,len_pdf))
    values[0,:] = per1
    values[1,:] = k1
    values[2,:] = secosw1
    values[3,:] = sesinw1
    values[4,:] = tc1

    # Define KDE
    kde = gaussian_kde(values, bw_method=None)
    kde_prior_obj = priors.KDEPrior(kde, total_params)

    for II in range(total_params):
        samples = kde_prior_obj.draw_samples(10000)
        assert np.mean(values[II,:]) == pytest.approx(np.mean(samples), abs=np.std(values[II,:]))

def test_ndinterpolator():
    # Read RV posteriors
    pdf_fromRadVel = pd.read_csv(orbitize.DATADIR+'sample_radvel_chains.csv.bz2', compression='bz2', index_col=0)
    per1 = pdf_fromRadVel.per1
    k1 = pdf_fromRadVel.k1 # Doppler semi-amplitude
    secosw1 = pdf_fromRadVel.secosw1
    sesinw1 = pdf_fromRadVel.sesinw1
    tc1 = pdf_fromRadVel.tc1
    
    # Put together posteriors to initialize ND interpolator
    len_pdf = len(pdf_fromRadVel)
    total_params = 5
    values = np.empty((total_params,len_pdf))
    values[0,:] = per1
    values[1,:] = k1
    values[2,:] = secosw1
    values[3,:] = sesinw1
    values[4,:] = tc1

    lnpriors_arr = pdf_fromRadVel.lnprobability.values
    
    # Define interp
    nearestNDinterp = NearestNDInterpolator(values.T,lnpriors_arr)
    nearestNDinterp_obj = priors.NearestNDInterpPrior(nearestNDinterp,total_params)

    for II in range(total_params):
        samples = nearestNDinterp_obj.draw_samples(10000)
        assert np.mean(values[II,:]) == pytest.approx(np.mean(samples), abs=np.std(values[II,:]))


if __name__=='__main__':
    test_ndinterpolator()
    test_kde()
    