"""Batch and scalar predictions must describe the same planetary systems."""
import os

import numpy as np
import pytest

import orbitize
from orbitize import read_input, sampler, system


@pytest.mark.parametrize("num_bodies", [2, 3])
@pytest.mark.parametrize("ordering", ["same", "mixed", "equal"])
@pytest.mark.parametrize("num_orbits", [1, 3])
def test_multiplanet_batch_predictions(num_bodies, ordering, num_orbits):
    """Preserve each sample's inner-companion selection. Written: Anton Soloviev."""
    data = read_input.read_file(os.path.join(orbitize.DATADIR, "test_val_multi.csv"))
    model = system.System(
        num_bodies, data, 1.47, 24.3, mass_err=0.11, plx_err=0.7,
        fit_secondary_mass=True, tau_ref_epoch=50000,
    )
    params = np.zeros((len(model.labels), num_orbits))
    for i, label in enumerate(model.labels):
        if label.startswith("sma"):
            params[i] = 80.0 - 15.0 * int(label[3:])
        elif label.startswith("ecc"):
            params[i] = 0.05
        elif label.startswith("inc"):
            params[i] = 0.6
        elif label.startswith("aop") or label.startswith("pan"):
            params[i] = 0.8
        elif label.startswith("tau"):
            params[i] = 0.2
        elif label == "plx":
            params[i] = 24.3
        elif label == "m0":
            params[i] = 1.47
        else:
            params[i] = 0.005
    if ordering == "mixed":
        params[model.param_idx["sma1"], -1] = 20.0
    elif ordering == "equal":
        params[model.sma_indx] = 50.0
    saved = params.copy()
    predictions, jitter = model.compute_model(params)
    if num_orbits == 1:
        predictions, jitter = predictions[..., None], jitter[..., None]
    mcmc = sampler.MCMC(model, num_walkers=64, num_threads=1)
    likelihoods = np.atleast_1d(mcmc._logl(params.T))
    for j in range(num_orbits):
        expected, expected_jitter = model.compute_model(params[:, j])
        np.testing.assert_allclose(predictions[..., j], expected, rtol=1e-13, atol=1e-11)
        np.testing.assert_array_equal(jitter[..., j], expected_jitter)
        np.testing.assert_allclose(likelihoods[j], mcmc._logl(params[:, j]), rtol=1e-12)
    np.testing.assert_array_equal(params, saved)
