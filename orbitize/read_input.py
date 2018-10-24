"""
Module to read user input from files and create standardized input for orbitize
"""

import numpy as np
from astropy.table import Table
from astropy.io.ascii import read, write

def read_formatted_file(filename):
    """ Reads data from any file
    readable by ``astropy.io.ascii.read()``, including csv format.
    See the `astropy docs <http://docs.astropy.org/en/stable/io/ascii/index.html#id1>`_.

    Here is an example of an orbitize-readable .csv input file::

        epoch,object,raoff,raoff_err,decoff,decoff_err,sep,sep_err,pa,pa_err,rv,rv_err
        1234,1,0.010,0.005,0.50,0.05,,,,,,
        1235,1,,,,,1.0,0.005,89.0,0.1,,
        1236,1,,,,,1.0,0.005,89.3,0.3,,
        1237,0,,,,,,,,,10,0.1

    Each row must have ``epoch`` (in MJD=JD-2400000.5) and ``object``. 
    Objects are numbered with integers, where the primary/central object is ``0``. 
    If you have, for example, one RV measurement of a star and three astrometric 
    measurements of an orbiting planet, you should put ``0`` in the ``object`` column 
    for the RV point, and ``1`` in the columns for the astrometric measurements. 

    Each line must also have at least one of the following sets of valid measurements:

        - RA and DEC offsets [mas], or
        - sep [mas] and PA [degrees East of NCP], or
        - RV measurement [km/s]

    .. Note:: Columns with no data can be omitted (e.g. if only separation and PA
        are given, the raoff, deoff, and rv columns can be excluded).

        If more than one valid set is given (e.g. RV measurement and astrometric measurement 
        taken at the same epoch), ``read_formatted_file()`` will generate a separate output 
        row for each valid set.

    .. Warning:: For now, ``orbitize`` only accepts astrometric measurements for one
        secondary body. In a future release, it will also handle astrometric measurements for
        multiple secondaries, RV measurements of the primary and secondar(ies), and astrometric
        measurements of the primary. Stay tuned!

    Args:
        filename (str): Input file name

    Returns:
        astropy.Table: Table containing orbitize-readable input for given
        object. For the example input above::

            epoch  object  quant1 quant1_err  quant2 quant2_err quant_type
           float64  int   float64  float64   float64  float64      str5
           ------- ------ ------- ---------- ------- ---------- ----------
           1234.0      1    0.01      0.005     0.5       0.05      radec
           1235.0      1     1.0      0.005    89.0        0.1      seppa
           1236.0      1     1.0      0.005    89.3        0.3      seppa
           1237.0      0    10.0        0.1     nan        nan         rv

        where ``quant_type`` is one of "radec", "seppa", or "rv".

        If ``quant_type`` is "radec" or "seppa", the units of quant are mas and degrees,
        if ``quant_type`` is "rv", the units of quant are km/s

    Written: Henry Ngo, 2018
    """

    # initialize output table
    output_table = Table(names=('epoch','object','quant1','quant1_err','quant2','quant2_err','quant_type'),
                         dtype=(float,int,float,float,float,float,'S5'))

    # read file
    input_table = read(filename)
    num_measurements = len(input_table)

    # validate input
    # if input_table is Masked, then figure out which entries are masked
    # otherwise, just check that we have the required columns
    if input_table.masked:
        if 'epoch' in input_table.columns:
            have_epoch = ~input_table['epoch'].mask
            if not have_epoch.all():
                raise Exception("Invalid input format: missing some epoch entries")
        else:
            raise Exception("Input table MUST have epoch!")
        if 'object' in input_table.columns:
            have_object = ~input_table['object'].mask
            if not have_object.all():
                raise Exception("Invalid input format: missing some object entries")
        else:
            raise Exception("Input table MUST have object id!")
        if 'raoff' in input_table.columns:
            have_ra = ~input_table['raoff'].mask
        else:
            have_ra = np.zeros(num_measurements, dtype=bool) # zeros are False
        if 'decoff' in input_table.columns:
            have_dec = ~input_table['decoff'].mask
        else:
            have_dec = np.zeros(num_measurements, dtype=bool) # zeros are False
        if 'sep' in input_table.columns:
            have_sep = ~input_table['sep'].mask
        else:
            have_sep = np.zeros(num_measurements, dtype=bool) # zeros are False
        if 'pa' in input_table.columns:
            have_pa = ~input_table['pa'].mask
        else:
            have_pa = np.zeros(num_measurements, dtype=bool) # zeros are False
        if 'rv' in input_table.columns:
            have_rv = ~input_table['rv'].mask
        else:
            have_rv = np.zeros(num_measurements, dtype=bool) # zeros are False
    else: # no masked entries, just check for required columns
        if 'epoch' not in input_table.columns:
            raise Exception("Input table MUST have epoch!")
        if 'object' not in input_table.columns:
            raise Exception("Input table MUST have object id!")
        if 'raoff' in input_table.columns:
            have_ra = np.ones(num_measurements, dtype=bool) # ones are False
        else:
            have_ra = np.zeros(num_measurements, dtype=bool) # zeros are False
        if 'decoff' in input_table.columns:
            have_dec = np.ones(num_measurements, dtype=bool) # ones are False
        else:
            have_dec = np.zeros(num_measurements, dtype=bool) # zeros are False
        if 'sep' in input_table.columns:
            have_sep = np.ones(num_measurements, dtype=bool) # ones are False
        else:
            have_sep = np.zeros(num_measurements, dtype=bool) # zeros are False
        if 'pa' in input_table.columns:
            have_pa = np.ones(num_measurements, dtype=bool) # ones are False
        else:
            have_pa = np.zeros(num_measurements, dtype=bool) # zeros are False
        if 'rv' in input_table.columns:
            have_rv = np.ones(num_measurements, dtype=bool) # ones are False
        else:
            have_rv = np.zeros(num_measurements, dtype=bool) # zeros are False

    # loop through each row and format table
    index=0
    for row in input_table:

        # check epoch format and put in MJD
        if row['epoch'] > 2400000.5: # assume this is in JD
            MJD = row['epoch'] - 2400000.5
        else:
            MJD = row['epoch']

        # check that "object" is an integer (instead of ABC/bcd)
        if not isinstance(row['object'], (int, np.int32, np.int64)):
            raise Exception("Invalid object ID. Object IDs must be integers.")

        # determine input quantity type (RA/DEC, SEP/PA, or RV)
        if have_ra[index] and have_dec[index]:
            output_table.add_row([MJD, row['object'], row['raoff'], row['raoff_err'], row['decoff'], row['decoff_err'], "radec"])
        elif have_sep[index] and have_pa[index]:
            output_table.add_row([MJD, row['object'], row['sep'], row['sep_err'], row['pa'], row['pa_err'], "seppa"])
        if have_rv[index]:
            output_table.add_row([MJD, row['object'], row['rv'], row['rv_err'], None, None, "rv"])
        index=index+1

    return output_table

def write_orbitize_input(table,output_filename,file_type='csv'):
    """ Writes orbitize-readable input as an ASCII file

    Args:
        table (astropy.Table): Table containing orbitize-readable input for given
            object, as generated by the read functions in this module.
        output_filename (str): csv file to write to
        file_type (str): Any valid write format for astropy.io.ascii. See the 
            `astropy docs <http://docs.astropy.org/en/stable/io/ascii/index.html#id1>`_.
            Defaults to csv.

    (written) Henry Ngo, 2018
    """

    # check format
    valid_formats = ['aastex', 'basic', 'commented_header', 'csv', 'ecsv',
                     'fixed_width', 'fixed_width_no_header', 'fixed_width_two_line',
                     'html', 'ipac', 'latex', 'no_header', 'rdb', 'rst', 'tab']
    if file_type not in valid_formats:
        raise Exception('Invalid output format specified.')

    # write file
    write(table,output=output_filename,format=file_type)

def read_orbitize_input(filename):
    """ Reads orbitize-readable input from a correctly formatted ASCII file

    Args:
        filename (str): Name of file to read. It should have columns
            indicated in the table below.

    Returns:
        astropy.Table: Table containing orbitize-readable input for given
        object. Columns returned are::

            epoch, object, quant1, quant1_err, quant2, quant2_err, quant_type

        where ``quant_type`` is one of "radec", "seppa", or "rv".

        If ``quant_type`` is "radec" or "seppa", the units of quant are mas and degrees,
        if ``quant_type`` is "rv", the units of quant are km/s

    (written) Henry Ngo, 2018
    """
    output_table=read(filename)
    return output_table
