"""
Module to read user input from files and create standardized input for orbitize
"""

import deprecation
import numpy as np
import orbitize
from astropy.table import Table
from astropy.io.ascii import read, write


def read_file(filename):
    """ Reads data from any file for use in orbitize
    readable by ``astropy.io.ascii.read()``, including csv format.
    See the `astropy docs <http://docs.astropy.org/en/stable/io/ascii/index.html#id1>`_.

    There are two ways to provide input data to orbitize.

    The first way is to provide astrometric measurements, shown with the following example.

    Example of an orbitize-readable .csv input file::

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
        taken at the same epoch), ``read_file()`` will generate a separate output row for
        each valid set.

    Alternatively, you can also supply a data file with the columns already corresponding to
    the orbitize format (see the example in description of what this method returns). This may
    be useful if you are wanting to use the output of the `write_orbitize_input` method.

    .. Note:: When providing data with columns in the orbitize format, there should be no
        empty cells. As in the example below, when quant2 is not applicable, the cell should
        contain nan.

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
    output_table = Table(names=('epoch', 'object', 'quant1', 'quant1_err', 'quant2', 'quant2_err', 'quant_type'),
                         dtype=(float, int, float, float, float, float, 'S5'))

    # read file
    try:
        input_table = read(filename)

        # convert to masked table
        if input_table.has_masked_columns:
            input_table = Table(input_table, masked=True, copy=False)

    except:
        raise Exception(
            'Unable to read file: {}. \n Please check file path and format.'.format(filename))
    num_measurements = len(input_table)

    # Decide if input was given in the orbitize style
    orbitize_style = 'quant_type' in input_table.columns

    # validate input
    # if input_table is Masked, then figure out which entries are masked
    # otherwise, just check that we have the required columns based on orbitize_style flag
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
        if orbitize_style:  # proper orbitize style should NEVER have masked entries (nan required)
            raise Exception("Input table in orbitize style may NOT have empty cells")
        else:  # Check for these things when not orbitize style
            if 'raoff' in input_table.columns:
                have_ra = ~input_table['raoff'].mask
            else:
                have_ra = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if 'decoff' in input_table.columns:
                have_dec = ~input_table['decoff'].mask
            else:
                have_dec = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if 'sep' in input_table.columns:
                have_sep = ~input_table['sep'].mask
            else:
                have_sep = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if 'pa' in input_table.columns:
                have_pa = ~input_table['pa'].mask
            else:
                have_pa = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if 'rv' in input_table.columns:
                have_rv = ~input_table['rv'].mask
            else:
                have_rv = np.zeros(num_measurements, dtype=bool)  # zeros are False
    else:  # no masked entries, just check for required columns
        if 'epoch' not in input_table.columns:
            raise Exception("Input table MUST have epoch!")
        if 'object' not in input_table.columns:
            raise Exception("Input table MUST have object id!")
        if not orbitize_style:  # Set these flags only when not already in orbitize style
            if 'raoff' in input_table.columns:
                have_ra = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_ra = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if 'decoff' in input_table.columns:
                have_dec = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_dec = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if 'sep' in input_table.columns:
                have_sep = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_sep = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if 'pa' in input_table.columns:
                have_pa = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_pa = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if 'rv' in input_table.columns:
                have_rv = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_rv = np.zeros(num_measurements, dtype=bool)  # zeros are False

    # loop through each row and format table
    index = 0
    for row in input_table:
        # First check if epoch is a number
        try:
            float_epoch = np.float(row['epoch'])
        except:
            raise Exception(
                'Problem reading epoch in the input file. Epoch should be given in MJD.')
        # check epoch format and put in MJD
        if row['epoch'] > 2400000.5:  # assume this is in JD
            print('Converting input epochs from JD to MJD.\n')
            MJD = row['epoch'] - 2400000.5
        else:
            MJD = row['epoch']

        # check that "object" is an integer (instead of ABC/bcd)
        if not isinstance(row['object'], (int, np.int32, np.int64)):
            raise Exception("Invalid object ID. Object IDs must be integers.")

        # determine input quantity type (RA/DEC, SEP/PA, or RV)
        if orbitize_style:
            if row['quant_type'] == 'rv':  # special format for rv rows
                output_table.add_row([MJD, row['object'], row['quant1'],
                                      row['quant1_err'], None, None, row['quant_type']])
            elif row['quant_type'] == 'radec' or row['quant_type'] == 'seppa':  # other allowed formats
                output_table.add_row([MJD, row['object'], row['quant1'], row['quant1_err'],
                                      row['quant2'], row['quant2_err'], row['quant_type']])
            else:  # catch wrong formats
                raise Exception("Invalid 'quant_type'. Valid values are 'radec', 'seppa' or 'rv'")
        else:  # When not in orbitize style
            if have_ra[index] and have_dec[index]:
                output_table.add_row([MJD, row['object'], row['raoff'],
                                      row['raoff_err'], row['decoff'], row['decoff_err'], "radec"])
            elif have_sep[index] and have_pa[index]:
                output_table.add_row([MJD, row['object'], row['sep'],
                                      row['sep_err'], row['pa'], row['pa_err'], "seppa"])
            if have_rv[index]:
                output_table.add_row([MJD, row['object'], row['rv'],
                                      row['rv_err'], None, None, "rv"])

        index = index+1

    return output_table


@deprecation.deprecated(deprecated_in="1.0.2", removed_in="2.0",
                        current_version=orbitize.__version__,
                        details="Use read_file() instead. v1.0.2 replaces read_formatted_file and read_orbitize_input with read_file(). For now, this will be a wrapper for read_file and will be removed in the v2.0 release.")
def read_formatted_file(filename):
    """
    Version 1.0.2 replaces this function with `read_file`.
    Currently exists as a wrapper for `read_file` and will be removed in v2.0

    Written: Henry Ngo, 2018
    """

    return read_file(filename)


def write_orbitize_input(table, output_filename, file_type='csv'):
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
    write(table, output=output_filename, format=file_type)


@deprecation.deprecated(deprecated_in="1.0.2", removed_in="2.0",
                        current_version=orbitize.__version__,
                        details="Use read_file() instead. v1.0.2 replaces read_orbitize_input and read_formatted_file with read_file(). For now, this will be a wrapper for read_file and will be removed in the v2.0 release.")
def read_orbitize_input(filename):
    """
    Version 1.0.2 replaces this function with `read_file`.
    Currently exists as a wrapper for `read_file` and will be removed in v2.0

    Written: Henry Ngo, 2018
    """
    return read_file(filename)
