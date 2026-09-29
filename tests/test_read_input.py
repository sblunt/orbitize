import pytest
import numpy as np
import os
import orbitize
from astropy import units as u
from astropy.table import Table
from orbitize.read_input import from_dl2, from_ohp_file, read_file, write_orbitize_input


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
    quant_type_expected = ["radec", "seppa", "seppa", "rv"]
    instrument_expected = ["defrd", "defsp", "defsp", "defrv"]
    assert len(input_table) == rows_expected
    for meas, truth in zip(input_table["epoch"], epoch_expected):
        assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table["object"], object_expected):
        assert truth == meas
    for meas, truth in zip(input_table["quant1"], quant1_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table["quant1_err"], quant1_err_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table["quant2"], quant2_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table["quant2_err"], quant2_err_expected):
        if np.isnan(truth):
            assert np.isnan(meas)
        else:
            assert truth == pytest.approx(meas)
    for meas, truth in zip(input_table["quant_type"], quant_type_expected):
        assert truth == meas

    for meas, truth in zip(input_table["instrument"], instrument_expected):
        assert truth == meas


def test_read_file():
    """
    Test the read_file function using the test_val.csv file and test_val_radec.csv
    """
    # Check that main test input is read in with correct values
    input_file = os.path.join(orbitize.DATADIR, "test_val.csv")
    _compare_table(read_file(input_file))
    # Check that an input value with all valid entries and only ra/dec
    # columns can be read
    input_file_radec = os.path.join(orbitize.DATADIR, "test_val_radec.csv")
    read_file(input_file_radec)


def test_write_orbitize_input():
    """
    Test the write_orbitize_input and the read_file functions
    """
    input_file = os.path.join(orbitize.DATADIR, "test_val.csv")
    test_table = read_file(input_file)
    output_file = os.path.join(orbitize.DATADIR, "temp_test_orbitize_input.csv")
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
    input_file = os.path.join(testdir, "test_val_cov.csv")
    input_data = read_file(input_file)
    _compare_table(input_data)
    print(input_data)
    # Check the covariance column
    quant12_corr_truth = [0.25, np.nan, -0.5, np.nan]
    assert "quant12_corr" in input_data.columns
    for row, truth in zip(input_data, quant12_corr_truth):
        meas = row["quant12_corr"]
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
    input_file = os.path.join(orbitize.DATADIR, "old_orbitize_format.csv")
    input_data = read_file(input_file)

    # check correlation and instrument are defualts
    assert np.isnan(input_data["quant12_corr"][0])
    assert input_data["instrument"][0] == "defsp"

    assert np.isnan(input_data["quant12_corr"][1])
    assert input_data["instrument"][1] == "defrd"

    assert np.isnan(input_data["quant12_corr"][2])
    assert input_data["instrument"][2] == "defrv"


def test_from_ohp_file():
    """
    Check that the OHP reader converts JD to MJD, preserves the measurements,
    and assigns the expected units.
    """

    # check reader and astropy match
    path = os.path.join(orbitize.DATADIR, "target_1.csv")
    expected = Table.read(path, format="ascii.csv")
    actual = from_ohp_file(path)
    expected["obs_time_tcb"] -= 2400000.5
    assert isinstance(actual, Table)
    assert len(actual) == 102
    assert actual.colnames == expected.colnames
    for name in expected.colnames:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert actual["obs_time_tcb"].unit == u.day
    assert actual["centroid_pos_al"].unit == u.mas
    assert actual["centroid_pos_error_al"].unit == u.mas
    assert actual["parallax_factor_al"].unit == u.dimensionless_unscaled
    assert actual["scan_pos_angle"].unit == u.rad
    assert actual.meta["time_format"] == "mjd"
    assert actual.meta["time_scale"] == "tcb"
    assert actual.meta["input_format"] == "ohp"
    unlabelled = actual.copy()
    unlabelled.meta.clear()
    with pytest.raises(ValueError, match="Expected raw OHP JD times"):
        from_ohp_file(unlabelled)


def test_from_ohp_file_missing_column():
    """
    Check that OHP reader is not missing required columns for orbit fitting.
    """

    path = os.path.join(orbitize.DATADIR, "target_1.csv")
    table = Table.read(path, format="ascii.csv")

    # remove a column, e.g. along scan position
    table.remove_column("centroid_pos_al")

    with pytest.raises(ValueError, match="Missing required OHP columns: centroid_pos_al"):
        from_ohp_file(table)


def test_from_ohp_file_empty():
    """
    Check that OHP reader is not reading an empty file

    """
    path = os.path.join(orbitize.DATADIR, "target_1.csv")
    table = Table.read(path, format="ascii.csv")[:0]
    with pytest.raises(ValueError, match="at least one observation"):
        from_ohp_file(table)


def test_from_dl2_mjd(tmp_path):
    """Convert native ns to barycentric MJD once, keeping separate CCDs."""
    native = Table({
        "source_id": [123], "transit_id": [0],
        "ra0": [60.0], "dec0": [-10.0], "parallax_factor_al": [0.5],
        "obs_time_bary_corr": [2_000_000_000],
        "obs_time_tcb": [86400_000_000_000 + np.arange(3) * 1_000_000_000],
        "centroid_pos_al": [[1.0, -10.0, 20.0]],
        "centroid_pos_error_al": [[0.1, 0.2, 0.3]],
        "scan_pos_angle": [[0.0, 90.0, 180.0]],
        "used_by_agis_al": [[False, True, True]],
    })
    path = tmp_path / "epoch_astrometry.xml"
    native.write(path, format="votable", tabledata_format="binary2")
    actual = from_dl2(path, source_id=123)
    np.testing.assert_allclose(
        actual["obs_time_tcb"], 55198.0 + np.array([3.0, 4.0]) / 86400,
        rtol=0, atol=1e-10,
    )
    np.testing.assert_array_equal(actual["component_index"], [1, 2])
    np.testing.assert_array_equal(actual["centroid_pos_al"], [-10.0, 20.0])
    np.testing.assert_allclose(actual["centroid_pos_error_al"], [0.2, 0.3])
    np.testing.assert_allclose(actual["scan_pos_angle"], [np.pi / 2, np.pi])
    assert actual.meta["time_format"] == "mjd"
    assert actual.meta["time_scale"] == "tcb"


if __name__ == "__main__":
    test_read_file()
    test_write_orbitize_input()
    test_cov_input()
    test_read_old_orbitize_format()
    test_from_ohp_file()
    test_from_ohp_file_missing_column()
    test_from_ohp_file_empty()
