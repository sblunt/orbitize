"""
Test the different Driver class creation options
"""

import pytest
import numpy as np
import orbitize
from orbitize import driver
from orbitize.read_input import read_file
import os

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

def test_create_driver_from_filename():
    """
    Test creation of Driver object from filename as input
    """
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    myDriver = driver.Driver(input_file, # path to data file
                             'MCMC', # name of algorith for orbit-fitting
                             1, # number of secondary bodies in system
                             1.0, # total system mass [M_sun]
                             50.0, # total parallax of system [mas]
                             mass_err=0.1, # mass error [M_sun]
                             plx_err=0.1, # parallax error [mas]
                             system_kwargs={'fit_secondary_mass':True}) 
    _compare_table(myDriver.system.data_table)


def test_create_driver_from_table():
    """
    Test creation of Driver object from Table as input
    """
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    input_table = read_file(input_file)
    myDriver = driver.Driver(input_table, # astropy.table Table of input
                             'MCMC', # name of algorithm for orbit-fitting
                             1, # number of secondary bodies in system
                             1.0, # total system mass [M_sun]
                             50.0, # total parallax of system [mas]
                             mass_err=0.1, # mass error [M_sun]
                             plx_err=0.1, # parallax error [mas]
                             system_kwargs={'fit_secondary_mass':True}) 
    _compare_table(myDriver.system.data_table)

def test_system_kwargs():
    """
    Test additional arguments to the system class
    """
    input_file = os.path.join(orbitize.DATADIR, 'test_val.csv')
    myDriver = driver.Driver(input_file, # path to data file
                             'MCMC', # name of algorith for orbit-fitting
                             1, # number of secondary bodies in system
                             1.0, # total system mass [M_sun]
                             50.0, # total parallax of system [mas]
                             mass_err=0.1, # mass error [M_sun]
                             plx_err=0.1, # parallax error [mas]
                             system_kwargs={"tau_ref_epoch": 50000, 'fit_secondary_mass':True}
    )
    assert myDriver.system.tau_ref_epoch == 50000

if __name__ == '__main__':
    test_create_driver_from_filename()
    test_create_driver_from_table()
    test_system_kwargs()
