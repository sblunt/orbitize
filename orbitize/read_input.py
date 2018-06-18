"""
Module to read user input from files and create standardized input for orbitize
"""

__author__ = 'Henry Ngo'

import numpy as np
from astropy.table import Table
from astropy.io.ascii import read, write

def read_formatted_file(filename):
    """Reads astrometric measurements for object from file in any format
    readable by ``astropy.io.ascii.read()``, including csv format.
    See `Astropy docs <http://docs.astropy.org/en/stable/io/ascii/index.html#id1>`_.

    The input file could have the headers in the CSV example below::

        epoch,object,raoff,raoff_err,decoff,decoff_err,sep,sep_err,pa,pa_err,rv,rv_err
        1234,1,0.010,0.005,0.50,0.05,,,,,,
        1235,1,,,,,1.0,0.005,89.0,0.1,,
        1236,1,,,,,1.0,0.005,89.3,0.3,,
        1237,0,,,,,,,,,10,0.1

    Each line must have ``epoch`` and ``object``. Objects are numbered with integers,
    where the primary/central object is ``0``.

    Each line must also have at least one of the following sets of valid measurements:
        - RA and DEC offsets (arcseconds), or
        - Sep (arcseconds) and PA (degrees), or
        - RV measurement (km/s)

    Note: Columns with no data can be omitted (e.g. if only separation and PA
    are given, the raoff, deoff, and rv columns can be excluded).

    If more than one valid set given, will generate separate output row for each valid set

    Args:
        filename (str): Input file name

    Returns:
        astropy.Table: Table containing orbitize-readable input for given
        object. Columns returned are shown in the example output below::

            epoch  object  quant1 quant1_err  quant2 quant2_err quant_type
           float64  int   float64  float64   float64  float64      str5
           ------- ------ ------- ---------- ------- ---------- ----------
           1234.0      1    0.01      0.005     0.5       0.05      radec
           1235.0      1     1.0      0.005    89.0        0.1      seppa
           1236.0      1     1.0      0.005    89.3        0.3      seppa
           1237.0      0    10.0        0.1     nan        nan         rv

        where ``quant_type`` is one of "radec", "seppa", or "rv". This example output corresponds to the example input shown above.

        If ``quant_type`` is "radec" or "seppa", the units of quant are arcseconds and degrees,
        if ``quant_type`` is "rv", the units of quant are km/s

    Written: Henry Ngo, 2018
    """

    # Initialize output table
    output_table = Table(names=('epoch','object','quant1','quant1_err','quant2','quant2_err','quant_type'),
                         dtype=(float,int,float,float,float,float,'S5'))

    # Read the CSV file
    input_table = read(filename)
    num_measurements = len(input_table)

    # Validate input
    if 'epoch' in input_table.columns:
        have_epoch = ~input_table['epoch'].mask
    else:
        raise Exception("Input table MUST have epoch!")
    if 'object' in input_table.columns:
        have_object = ~input_table['object'].mask
    else:
        raise Exception("Input table MUST have object id!")
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
    if not have_object.all():
        raise Exception("Invalid input format: missing some object entries")

    # Loop through each row and format table
    index=0
    for row in input_table:
        # Check epoch format and puts in MJD (MJD = JD - 2400000.5)
        if row['epoch'] > 2400000.5: # Assume this is in JD
            MJD = row['epoch'] - 2400000.5
        else:
            MJD = row['epoch']
        # Check that "object" is an integer (instead of ABC/bcd)
        if not isinstance(row['object'], (int, np.int32, np.int64)):
            raise Exception("Invalid object ID. Object IDs must be integers.")
        # Determine input quantity type (RA/DEC, SEP/PA, or RV?)
        if have_ra[index] and have_dec[index]:
            output_table.add_row([MJD, row['object'], row['raoff'], row['raoff_err'], row['decoff'], row['decoff_err'], "radec"])
        if have_sep[index] and have_pa[index]:
            output_table.add_row([MJD, row['object'], row['sep'], row['sep_err'], row['pa'], row['pa_err'], "seppa"])
        if have_rv[index]:
            output_table.add_row([MJD, row['object'], row['rv'], row['rv_err'], None, None, "rv"])
        index=index+1

    return output_table

def write_orbitize_input(table,output_filename,file_type='csv'):
    """Writes orbitize-readable input as an ASCII file

    Args:
        table (astropy.Table): Table containing orbitize-readable input for given
        object, as generated by the read functions in this module.
        output_filename (str): Name of output csv file to write
        file_type (str): Any valid write format for astropy.io.ascii
        See: `http://docs.astropy.org/en/stable/io/ascii/index.html#id1 <http://docs.astropy.org/en/stable/io/ascii/index.html#id1>`_
        Defaults to csv.

    Returns:
        Nothing

    (written) Henry Ngo, 2018
    """
    # Check format
    valid_formats = ['aastex', 'basic', 'commented_header', 'csv', 'ecsv',
                     'fixed_width', 'fixed_width_no_header', 'fixed_width_two_line',
                     'html', 'ipac', 'latex', 'no_header', 'rdb', 'rst', 'tab']
    if file_type not in valid_formats:
        raise Exception('Invalid output format specified.')

    # Write file
    write(table,output=output_filename,format=file_type)

def read_orbitize_input(filename):
    """Reads orbitize-readable input from a correctly formatted ASCII file

    Args:
        filename (str): Name of file to read. It should have the same columns
        indicated in the table below.

    Returns:
        astropy.Table: Table containing orbitize-readable input for given
        object. Columns returned are::

            epoch, object, quant1, quant1_err, quant2, quant2_err, quant_type

        where ``quant_type`` is one of "radec", "seppa", or "rv".

        If ``quant_type`` is "radec" or "seppa", the units of quant are arcseconds and degrees,
        if ``quant_type`` is "rv", the units of quant are km/s

    (written) Henry Ngo, 2018
    """
    output_table=read(filename)
    return output_table
