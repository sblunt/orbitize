"""
Module to read user input and create standardized input for orbitize
"""

__author__ = 'Henry Ngo'
__all__ = ['read_csv']

import numpy as np
from astropy.table import Table
from astropy.io.ascii import read

def read_csv(filename):
    """Reads astrometric measurements for object from file in CSV format

    Input file can have columns (okay to omit columns with no data):
    epoch,raoff,raoff_err,decoff,decoff_err,sep,sep_err,pa,pa_err,rv,rv_err

    Generally expect to receive one of the following sets of valid measurements:
    - RA and DEC offset, or
    - Sep and PA, or
    - RV measurement
    If more than one valid set given, will generate separate output row for each valid set

    Args:
        filename (str): Input file name

    Returns:
        Astropy Table containing all astrometric measurements for given object with columns:
        epoch,quant1,quant1_err,quant2,quant2_err,quant_type
        where quant_type is one of "radec", "seppa", or "rv"
    """

    # Initialize output table
    output_table = Table(names=('epoch','quant1','quant1_err','quant2','quant2_err','quant_type'),
                         dtype=(float,float,float,float,float,'S5'))

    # Read the CSV file
    input_table = read(filename)
    num_measurements = len(input_table)

    # Validate input
    if 'epoch' in input_table.columns:
        have_epoch = ~input_table['epoch'].mask
    else:
        raise Exception("Input table MUST have epoch!")
    if 'raoff' in input_table.columns:
        have_ra = ~input_table['raoff'].mask
    else:
        have_ra = np.zeros(num_measurements, dtype=bool) # Zeros are False
    if 'decoff' in input_table.columns:
        have_dec = ~input_table['decoff'].mask
    else:
        have_dec = np.zeros(num_measurements, dtype=bool) # Zeros are False
    if 'sep' in input_table.columns:
        have_sep = ~input_table['sep'].mask
    else:
        have_sep = np.zeros(num_measurements, dtype=bool) # Zeros are False
    if 'pa' in input_table.columns:
        have_pa = ~input_table['pa'].mask
    else:
        have_pa = np.zeros(num_measurements, dtype=bool) # Zeros are False
    if 'rv' in input_table.columns:
        have_rv = ~input_table['rv'].mask
    else:
        have_rv = np.zeros(num_measurements, dtype=bool) # Zeros are False

    if not have_epoch.all():
        raise Exception("Invalid input format: missing some epoch entries")

    # Loop through each row and format table
    index=0
    for row in input_table:
        # Check epoch format and puts in MJD (MJD = JD - 2400000.5)
        if row['epoch'] > 2400000.5: # Assume this is in JD
            MJD = row['epoch'] - 2400000.5
        else:
            MJD = row['epoch']
        # Determine input quantity type (RA/DEC, SEP/PA, or RV?)
        if have_ra[index] and have_dec[index]:
            output_table.add_row([MJD, row['raoff'], row['raoff_err'], row['decoff'], row['decoff_err'], "radec"])
        if have_sep[index] and have_pa[index]:
            output_table.add_row([MJD, row['sep'], row['sep_err'], row['pa'], row['pa_err'], "seppa"])
        if have_rv[index]:
            output_table.add_row([MJD, row['rv'], row['rv_err'], None, None, "rv"])
        index=index+1

    return output_table
