import numpy as np
import pandas as pd
import os
from astropy.time import Time
from pytest import approx

import orbitize
from orbitize.radvel_utils.compute_sep import compute_sep

def test_compute_sep():

    input_file = os.path.join(orbitize.DATADIR, 'sample_radvel_chains.csv.bz2')

    df = pd.read_csv(input_file, index_col=0)
    epochs = Time([2022, 2018], format='decimalyear')

    seps, df_orb = compute_sep(
        df, epochs, 'per tc secosw sesinw k', 0.82, 0.02, 312.22, 0.47
    )

    # test that the average inclination is 90 deg
    assert np.median(df_orb['inc_rad']) == approx(np.pi / 2, abs=0.2)

    # test that the seps output has the expected shape
    assert seps.shape == (2, 1000)

    # test that the seps are consistent with predictions from indep. code (by Lea Hirsch)
    assert np.median(seps[0]) == approx(1200, abs=200)
    assert np.median(seps[1]) == approx(900, abs=200)

if __name__=='__main__':
    test_compute_sep()