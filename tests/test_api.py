"""
Test the functionality of the API
"""
import numpy as np
import orbitize
import orbitize.lnlike as lnlike
import orbitize.system as system
import orbitize.sampler as sampler
import orbitize.read_input as read_input
import os


def test_compute_model():
    """
    Test basic functionality of ``System.compute_model()``
    """
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    data_table = read_input.read_file(input_file)
    data_table['object'] = 1
    testSystem_parsing = system.System(
        1, data_table, 10., 10.
    )

    params_arr = np.array([[1., 0.5], [0., 0.], [0., 0.], [0., 0.], [
                          0., 0.], [245000., 245000.], [10, 10], [10, 10]])
    model, jitter = testSystem_parsing.compute_model(params_arr)
    assert model.shape == (4, 2, 2)

    params_arr = np.array([1., 0., 0., 0., 0., 245000., 10, 10])
    model, jitter = testSystem_parsing.compute_model(params_arr)
    assert model.shape == (4, 2)


def test_systeminit():
    """
    Test that initializing a ``System`` class produces a list of ``Prior``
    objects of the correct length when:
        - parallax and total mass are fixed
        - parallax and total mass errors are given
        - parallax is fixed, total mass error is given
        - parallax error is given, total mass error is fixed

    Test that the different types of data are parsed correctly
    when initializing a ``System`` object.
    """
    testdir = orbitize.DATADIR
    input_file = os.path.join(testdir, 'test_val.csv')
    data_table = read_input.read_file(input_file)

    # Manually set 'object' column of data table
    data_table['object'] = 1
    data_table['object'][1] = 2

    plx_mass_errs2lens = {
        (0., 0.): 14,
        (1., 1.): 14,
        (0., 1.): 14,
        (1., 0.): 14
    }

    for plx_e, mass_e in plx_mass_errs2lens.keys():

        testSystem_priors = system.System(
            2, data_table, 10., 10., plx_err=plx_e, mass_err=mass_e
        )
        assert len(testSystem_priors.sys_priors) == \
            plx_mass_errs2lens[(plx_e, mass_e)]

    testSystem_parsing = system.System(
        2, data_table, 10., 10.,
        plx_err=0.5, mass_err=0.5
    )
    assert len(data_table[testSystem_parsing.seppa[0]]) == 0
    assert len(data_table[testSystem_parsing.seppa[1]]) == 1
    assert len(data_table[testSystem_parsing.seppa[2]]) == 1
    assert len(data_table[testSystem_parsing.radec[0]]) == 0
    assert len(data_table[testSystem_parsing.radec[1]]) == 1
    assert len(data_table[testSystem_parsing.radec[2]]) == 0

    assert testSystem_parsing.labels == [
        'sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'sma2',
        'ecc2', 'inc2', 'aop2', 'pan2', 'tau2', 'plx', 'mtot'
    ]


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

    chi2 = lnlike.chi2_lnlike(data, errors, model, jitter, seppa_indices)
    assert chi2.shape == (3, 2)
    assert (chi2 == -0.5 * np.ones((3, 2)) - np.log(np.sqrt(2*np.pi*np.ones((3, 2))))).all()

    # test with multiple models
    model = np.zeros((3, 2, 5))
    jitter = np.zeros((3, 2, 5))
    data = np.ones((3, 2))
    errors = np.ones((3, 2))

    seppa_indices = [np.array([1])]

    chi2 = lnlike.chi2_lnlike(data, errors, model, jitter, seppa_indices)
    assert chi2.shape == (3, 2, 5)
    assert (chi2 == -0.5 * np.ones((3, 2, 5)) - np.log(np.sqrt(2*np.pi*np.ones((3, 2, 5))))).all()


def test_custom_likelihood():
    """
    Tests the inclusion of a custom likelihood function in the code
    """
    # use the test_csv dir
    testdir = orbitize.DATADIR
    input_file = os.path.join(testdir, 'GJ504.csv')
    data_table = read_input.read_file(input_file)
    # Manually set 'object' column of data table
    data_table['object'] = 1

    # construct the system
    orbit = system.System(1, data_table, 1, 0.01)

    # construct custom likelihood function
    def my_likelihood(params):
        return -5

    # construct sampler
    n_walkers = 100
    mcmc1 = sampler.MCMC(orbit, 0, n_walkers, num_threads=1)
    mcmc2 = sampler.MCMC(orbit, 0, n_walkers, num_threads=1, custom_lnlike=my_likelihood)

    param = np.array([2, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.01])

    logl1 = mcmc1._logl(param)
    logl2 = mcmc2._logl(param)

    assert logl1 == logl2 + 5


def test_radec2seppa():
    """
    Basic test for convenience function converting RA/DEC to SEP/PA
    """
    ra = np.array([-1., 0., -1., 1.])
    dec = np.array([0., -1., -1., 1.])
    sep, pa = system.radec2seppa(ra, dec)
    assert sep.all() == np.array([1., 1., np.sqrt(2.), np.sqrt(2.)]).all()
    assert pa.all() == np.array([270., 180., 225., 45.]).all()


if __name__ == "__main__":
    test_custom_likelihood()
