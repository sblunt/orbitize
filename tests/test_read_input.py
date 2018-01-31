import pytest
import numpy as np
import os
from orbitize.read_input import read_formatted_file


def _compare_table(input_table):
    """
    Tests input table to expected values, which are:
      epoch   quant1 quant1_err  quant2 quant2_err quant_type
     float64 float64  float64   float64  float64      str5
     ------- ------- ---------- ------- ---------- ----------
     1234.0    0.01      0.005     0.5       0.05      radec
     1235.0     1.0      0.005    89.0        0.1      seppa
     1236.0     1.0      0.005    89.3        0.3      seppa
     1236.0    10.0        0.1     nan        nan         rv
    """
    rows_expected = 4
    epoch_expected = [1234, 1235, 1236, 1236]
    quant1_expected = [0.01, 1.0, 1.0, 10.0]
    quant1_err_expected = [0.005, 0.005, 0.005, 0.1]
    quant2_expected = [0.5, 89.0, 89.3, np.nan]
    quant2_err_expected = [0.05, 0.1, 0.3, np.nan]
    quant_type_expected = ['radec', 'seppa', 'seppa', 'rv']
    assert len(input_table) == rows_expected
    for meas,truth in zip(input_table['epoch'],epoch_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas,truth in zip(input_table['quant1'],quant1_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas,truth in zip(input_table['quant1_err'],quant1_err_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas,truth in zip(input_table['quant2'],quant2_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas,truth in zip(input_table['quant2_err'],quant2_err_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas,truth in zip(input_table['quant_type'],quant_type_expected):
            assert truth == meas

def test_read_formatted_file():
    """
    Test the read_formatted_file function using the test_val.csv file
    """
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    _compare_table(read_formatted_file(input_file))

def test_read_write_orbitize_input():
    """
    Test the read_orbitize_input and the write_orbitize_input functions
    """

if __name__ == "__main__":
    test_read_formatted_file()
