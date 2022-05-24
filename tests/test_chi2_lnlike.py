import orbitize.driver 
import numpy as np
import orbitize.lnlike as lnlike
import pytest

def test_chi2lnlike():
    """
    Test the ability of ``orbitize.lnlike.chi2_lnlike()``
    to work properly on arrays.
    """
    # test with a single model
    model = np.zeros((3, 2))
    jitter = np.zeros((3, 2))
    data = np.ones((3, 2))
    errors = np.ones((3, 2))

    seppa_indices = [np.array([1])]

    chi2 = lnlike.chi2_lnlike(data, errors, None, model, jitter, seppa_indices)
    assert chi2.shape == (3, 2)
    assert chi2 == pytest.approx(
        -0.5 * np.ones((3, 2)) - np.log(np.sqrt(2*np.pi*np.ones((3, 2))))
    )

    # test with multiple models
    model = np.zeros((3, 2, 5))
    jitter = np.zeros((3, 2, 5))
    data = np.ones((3, 2))
    errors = np.ones((3, 2))

    seppa_indices = [np.array([1])]

    chi2 = lnlike.chi2_lnlike(data, errors, None, model, jitter, seppa_indices)
    assert chi2.shape == (3, 2, 5)
    assert chi2 == pytest.approx(
        -0.5 * np.ones((3, 2, 5)) - np.log(np.sqrt(2*np.pi*np.ones((3, 2, 5))))
    )


def test_chi2lnlike_withcov():
    """
    Test the ability of ``orbitize.lnlike.chi2_lnlike()`` to work with some or all data having covariances
    """
    ### all all covariances
    data = np.array([[5,-4], [3,-2], [1,0] ])
    model = np.zeros(data.shape)
    jitter = np.zeros(data.shape)
    errs = np.array([[2,2], [2,2], [2,2]])
    covs = np.array([1, 0.25, 0.25])
    corrs = covs/errs[:,0]/errs[:,1]

    chi2s = lnlike.chi2_lnlike(data, errs, corrs, model, jitter, [])

    residuals = data - model
    for res, err, cov, chi2 in zip(residuals, errs, covs, chi2s):
        cov_matrix = np.array([[err[0]**2, cov], [cov, err[1]**2]])
        cov_inv = np.linalg.inv(cov_matrix)
        cov_inv_dot_diff = np.dot(cov_inv, res)
        logdet = np.linalg.slogdet(cov_matrix)[1]
        res_cov_res = res.dot(cov_inv_dot_diff)
        numpy_chi2 = -0.5 * (res_cov_res + logdet + 2 * np.log(2 * np.pi)) 

        assert np.sum(chi2) == pytest.approx(numpy_chi2)

    ### only one covariance term
    covs = np.array([1, np.nan, np.nan])
    corrs = covs/errs[:,0]/errs[:,1]
    new_chi2s = lnlike.chi2_lnlike(data, errs, corrs, model, jitter, [])

    assert chi2s[0] == pytest.approx(new_chi2s[0])


def test_2x2_analytical_solution():
    """
    Tests that our analytical solution to the 2x2 covariance matrix is correct
    """
    residuals = np.array([[5,-4], [3,-2], [1,0] ])

    errs = np.array([[2,2], [2,2], [2,2]])
    covs = np.array([1, 0.25, 0.25])
    corrs = covs/errs[:,0]/errs[:,1]

    chi2s = lnlike._chi2_2x2cov(np.array([residuals]), np.array([errs**2]), corrs)

    # compare to numpy solution
    for res, err, cov, chi2 in zip(residuals, errs, covs, chi2s[0]):
        cov_matrix = np.array([[err[0]**2, cov], [cov, err[1]**2]])
        cov_inv = np.linalg.inv(cov_matrix)
        cov_inv_dot_diff = np.dot(cov_inv, res)
        logdet = np.linalg.slogdet(cov_matrix)[1]
        res_cov_res = res.dot(cov_inv_dot_diff)
        numpy_chi2 = -0.5 * (res_cov_res + logdet + 2 * np.log(2 * np.pi)) 

        assert np.sum(chi2) == pytest.approx(numpy_chi2)


def test_chi2_log():
    #initiate OFTI driver with chi2 log
    myDriver = orbitize.driver.Driver(
    '{}/GJ504.csv'.format(orbitize.DATADIR), 'OFTI', 1, 1.22, 56.95, mass_err=0.08, plx_err=0.26, chi2_type='log')
    s = myDriver.sampler
    params = [44, 0, 45*np.pi/180, 0, 325*np.pi/180, 0, 56.95, 1.22]
    log_chi2 = s._logl(params)

    sys = myDriver.system
    data = np.array([sys.data_table['quant1'], sys.data_table['quant2']]).T
    errors = np.array([sys.data_table['quant1_err'], sys.data_table['quant2_err']]).T
    model, jitter = sys.compute_model(params)

    sep_data = data[:,0]
    sep_model = model[:, 0]
    sep_error = errors[:,0]
    pa_data = data[:,1]
    pa_model = model[:, 1]
    pa_error = errors[:,1]*np.pi/180


    #calculating sep chi squared
    sep_chi2_log = (np.log(sep_data)-np.log(sep_model))**2/(sep_error/sep_data)**2

    #calculting pa chi squared Log
    pa_resid = (pa_model-pa_data +180.) % 360. - 180.
    pa_chi2_log = 2*(1-np.cos(pa_resid*np.pi/180))/pa_error**2

    chi2 = np.zeros((len(sep_data),2))

    sigma2 = errors**2 + jitter**2 

    chi2[:,0] = sep_chi2_log
    chi2[:,1] = pa_chi2_log

    chi2 = -0.5 * chi2 - np.log(np.sqrt(2*np.pi*sigma2))

    lnlike = np.sum(chi2)

    assert lnlike == pytest.approx(log_chi2)


def test_log_vs_standard():
    
    #initiate driver with standard chi2
    myDriver_standard = orbitize.driver.Driver('{}/GJ504.csv'.format(orbitize.DATADIR),'OFTI', 1, 1.22, 56.95, mass_err=0.08, plx_err=0.26)
    s_standard = myDriver_standard.sampler
    orbits = s_standard.run_sampler(3000)

    #initiate driver with log chi2
    myDriver_log = orbitize.driver.Driver('{}/GJ504.csv'.format(orbitize.DATADIR), 'OFTI', 1, 1.22, 56.95, mass_err=0.08, plx_err=0.26, chi2_type = 'log')
    s_log = myDriver_log.sampler
    orbits = s_log.run_sampler(3000)   

    #take mean of result objects
    myResults_standard = np.mean(s_standard.results.post,axis=0)
    myResults_log = np.mean(s_log.results.post,axis=0)

    assert myResults_log == pytest.approx(myResults_standard, rel=0.05) 


if __name__ == "__main__":
    test_chi2lnlike()
    test_chi2_log()
    test_chi2lnlike_withcov()
    test_chi2lnlike_withcov()
    test_2x2_analytical_solution()

