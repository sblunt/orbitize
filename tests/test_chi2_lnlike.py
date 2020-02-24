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

def test_2x2_analytical_solution():
    """
    Tests that our analytical solution to the 2x2 covariance matrix is correct
    """
    residuals = np.array([[5,-4], [3,-2], [1,0] ])

    errs = np.array([[2,2], [2,2], [2,2]])
    covs = np.array([1, 0.25, 0.25])

    chi2 = lnlike._chi2_2x2cov(residuals, errs, covs)

    # compare to numpy solution


if __name__ == "__main__":
    test_chi2lnlike()
    test_2x2_analytical_solution()

