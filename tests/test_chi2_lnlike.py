import numpy as np
import orbitize.lnlike as lnlike

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
    assert (chi2 == -0.5 * np.ones((3, 2)) - np.log(np.sqrt(2*np.pi*np.ones((3, 2))))).all()

    # test with multiple models
    model = np.zeros((3, 2, 5))
    jitter = np.zeros((3, 2, 5))
    data = np.ones((3, 2))
    errors = np.ones((3, 2))

    seppa_indices = [np.array([1])]

    chi2 = lnlike.chi2_lnlike(data, errors, None, model, jitter, seppa_indices)
    assert chi2.shape == (3, 2, 5)
    assert (chi2 == -0.5 * np.ones((3, 2, 5)) - np.log(np.sqrt(2*np.pi*np.ones((3, 2, 5))))).all()


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

        assert np.sum(chi2) == numpy_chi2

    ### only one covariance term
    covs = np.array([1, np.nan, np.nan])
    corrs = covs/errs[:,0]/errs[:,1]
    new_chi2s = lnlike.chi2_lnlike(data, errs, corrs, model, jitter, [])

    assert np.all(chi2s[0] == new_chi2s[0])


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

        assert np.sum(chi2) == numpy_chi2


if __name__ == "__main__":
    test_chi2lnlike_withcov()
    test_2x2_analytical_solution()

