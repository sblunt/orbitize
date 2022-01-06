import pytest
import numpy as np
import os
import orbitize
from orbitize.read_input import read_file, write_orbitize_input


def _compare_table(input_table):
    """
    Tests input table to expected values, which are:
        epoch  object  quant1 quant1_err  quant2 quant2_err quant_type
       float64  int   float64  float64   float64  float64      str5
       ------- ------ ------- ---------- ------- ---------- ----------
       1234.0      1    0.01      0.005     0.5       0.05      radec
       1235.0      1     1.0      0.005    89.0        0.1      seppa
       1236.0      1     1.0      0.005    89.3        0.3      seppa
       1237.0      0    10.0        0.1     nan        nan         rv
    """
    rows_expected = 4
    epoch_expected = [1234, 1235, 1236, 1237]
    object_expected = [1, 1, 1, 0]
    quant1_expected = [0.01, 1.0, 1.0, 10.0]
    quant1_err_expected = [0.005, 0.005, 0.005, 0.1]
    quant2_expected = [0.5, 89.0, 89.3, np.nan]
    quant2_err_expected = [0.05, 0.1, 0.3, np.nan]
    quant_type_expected = ['radec', 'seppa', 'seppa', 'rv']
    instrument_expected = ['defrd', 'defsp', 'defsp', 'defrv']
    assert len(input_table) == rows_expected
    for meas, truth in zip(input_table['epoch'], epoch_expected):
        assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table['object'], object_expected):
        assert truth == meas
    for meas, truth in zip(input_table['quant1'], quant1_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table['quant1_err'], quant1_err_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table['quant2'], quant2_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table['quant2_err'], quant2_err_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table['quant_type'], quant_type_expected):
        assert truth == meas

    for meas, truth in zip(input_table['instrument'], instrument_expected):
        assert truth == meas


def test_read_file():
    """
    Test the read_file function using the test_val.csv file and test_val_radec.csv
    """
    # Check that main test input is read in with correct values
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    _compare_table(read_file(input_file))
    # Check that an input value with all valid entries and only ra/dec columns can be read
    input_file_radec = os.path.join(orbitize.DATADIR, 'test_val_radec.csv')
    read_file(input_file_radec)


def test_write_orbitize_input():
    """
    Test the write_orbitize_input and the read_file functions
    """
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    test_table = read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, 'temp_test_orbitize_input.csv')
    # If temp output file already exists, delete it
    if os.path.isfile(output_file):
        os.remove(output_file)
    try:  # Catch these tests so that we remove temporary file
        # Test that we were able to write the table
        write_orbitize_input(test_table, output_file)
        assert os.path.isfile(output_file)
        # Test that we can read the table and check if it's correct
        test_table_2 = read_file(output_file)
        _compare_table(test_table_2)
    finally:
        # Remove temporary file
        os.remove(output_file)


def test_cov_input():
    """
    Test including radec and seppa covariances/correlations.
    """
    testdir = orbitize.DATADIR
    # Check that main test input is read in with correct values
    input_file = os.path.join(testdir, 'test_val_cov.csv')
    input_data = read_file(input_file)
    _compare_table(input_data)
    print(input_data)
    # Check the covariance column
    quant12_corr_truth = [0.25, np.nan, -0.5, np.nan]
    assert 'quant12_corr' in input_data.columns
    for row, truth in zip(input_data, quant12_corr_truth):
        meas = row['quant12_corr']
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)

def test_read_old_orbitize_format():
    """
    Test the read_file function when using an old orbitize data file without 
    `quant12_corr` and `instrument` fields. 
    """
    # Check that main test input is read in with correct values
    input_file = os.path.join(orbitize.DATADIR, 'old_orbitize_format.csv')
    input_data = read_file(input_file)
    
    # check correlation and instrument are defualts
    assert np.isnan(input_data['quant12_corr'][0])
    assert input_data['instrument'][0] == 'defsp'

    assert np.isnan(input_data['quant12_corr'][1])
    assert input_data['instrument'][1] == 'defrd'

    assert np.isnan(input_data['quant12_corr'][2])
    assert input_data['instrument'][2] == 'defrv'



if __name__ == "__main__":
    test_read_file()
    test_write_orbitize_input()
    test_cov_input()
    test_read_old_orbitize_format()
