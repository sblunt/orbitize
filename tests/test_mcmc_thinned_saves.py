"""Keep real thinned MCMC samples in periodic result files."""
import os

import numpy as np
import pytest

import orbitize
from orbitize.driver import Driver
from orbitize.results import Results


@pytest.mark.parametrize("steps", [11, 12])
@pytest.mark.parametrize("num_temps", [1, 2])
@pytest.mark.parametrize("thin", [1, 2, 3])
@pytest.mark.parametrize("frequency", [2, 5])
def test_thinned_periodic_save(tmp_path, monkeypatch, steps, num_temps, thin, frequency):
    """Compare saved sample/likelihood pairs with the live backend.

    Written: Anton Soloviev, 2026
    """
    np.random.seed(394)
    driver = Driver(
        os.path.join(orbitize.DATADIR, "GJ504.csv"), "MCMC", 1, 1.22, 56.95,
        mcmc_kwargs={"num_temps": num_temps, "num_walkers": 16, "num_threads": 1},
    )
    prefix = str(tmp_path / "chain")
    checkpoints = []
    save_results = Results.save_results

    def save_and_reload(self, filename):
        """Read each real checkpoint. Written: Anton Soloviev, 2026."""
        save_results(self, filename)
        checkpoint = Results()
        checkpoint.load_results(filename)
        checkpoints.append(checkpoint)

    monkeypatch.setattr(Results, "save_results", save_and_reload)
    backend = driver.sampler.run_sampler(
        16 * steps, thin=thin, output_filename=prefix, periodic_save_freq=frequency
    )
    loaded = Results()
    loaded.load_results(prefix + ".hdf5")
    chain = backend.chain[0].reshape(-1, driver.sampler.num_params)
    expected = np.column_stack((
        chain, np.full(len(chain), 56.95), np.full(len(chain), 1.22),
        backend.loglikelihood[0].flatten(),
    ))
    actual = np.column_stack((loaded.post, loaded.lnlike))
    assert sorted(map(tuple, actual)) == sorted(map(tuple, expected))
    np.testing.assert_array_equal(loaded.post, driver.sampler.results.post)
    np.testing.assert_array_equal(loaded.curr_pos, driver.sampler.curr_pos)
    for index, checkpoint in enumerate(checkpoints[:-1], start=1):
        stored = index * frequency // thin
        chain = backend.chain[0, :, :stored].reshape(-1, driver.sampler.num_params)
        if stored == 0:
            assert checkpoint.post is None
            continue
        expected = np.column_stack((
            chain, np.full(len(chain), 56.95), np.full(len(chain), 1.22),
            backend.loglikelihood[0, :, :stored].flatten(),
        ))
        actual = np.column_stack((checkpoint.post, checkpoint.lnlike))
        assert sorted(map(tuple, actual)) == sorted(map(tuple, expected))
