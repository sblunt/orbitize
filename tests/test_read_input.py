import pytest
import deprecation
import numpy as np
import os
import orbitize
from orbitize.read_input import read_file, write_orbitize_input, read_formatted_file, read_orbitize_input


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
    object_expected = [1,1,1,0]
    quant1_expected = [0.01, 1.0, 1.0, 10.0]
    quant1_err_expected = [0.005, 0.005, 0.005, 0.1]
    quant2_expected = [0.5, 89.0, 89.3, np.nan]
    quant2_err_expected = [0.05, 0.1, 0.3, np.nan]
    quant_type_expected = ['radec', 'seppa', 'seppa', 'rv']
    assert len(input_table) == rows_expected
    for meas,truth in zip(input_table['epoch'],epoch_expected):
        assert truth == pytest.approx(meas)
    for meas,truth in zip(input_table['object'],object_expected):
        assert truth == meas
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

@deprecation.fail_if_not_removed
def test_read_formatted_file():
    """
    Tests the read_formatted_file function using the test_val.csv file and test_val_radec.csv

    This test exists with the fail_if_not_removed decorator as a reminder to remove in v2.0
    """
    # Check that main test input is read in with correct values
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    _compare_table(read_formatted_file(input_file))
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
        write_orbitize_input(test_table,output_file)
        assert os.path.isfile(output_file)
        # Test that we can read the table and check if it's correct
        test_table_2 = read_file(output_file)
        _compare_table(test_table_2)
    finally:
        # Remove temporary file
        os.remove(output_file)

@deprecation.fail_if_not_removed
def test_write_orbitize_input_2():
    """
    Test the write_orbitize_input and the read_orbitize_input functions

    This test exists with the fail_if_not_removed decorator as a reminder to remove in v2.0
    """
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    test_table = read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, 'temp_test_orbitize_input.csv')
    # If temp output file already exists, delete it
    if os.path.isfile(output_file):
        os.remove(output_file)
    try:  # Catch these tests so that we remove temporary file
        # Test that we were able to write the table
        write_orbitize_input(test_table,output_file)
        assert os.path.isfile(output_file)
        # Test that we can read the table and check if it's correct
        test_table_2 = read_orbitize_input(output_file)
        _compare_table(test_table_2)
    finally:
        # Remove temporary file
        os.remove(output_file)

if __name__ == "__main__":
    test_read_file()
    test_read_formatted_file()
    test_write_orbitize_input()
    test_write_orbitize_input_2()
