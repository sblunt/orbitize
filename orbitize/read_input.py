"""
Module to read user input from files and create standardized input for orbitize
"""

import numpy as np
import orbitize
from astropy.table import Table
from astropy.io.ascii import read, write

VALID_ATTRS = ["raoff", "decoff", "radec_corr", "sep", "pa", "seppa_corr", "rv", "brightness", "instrument"]

def read_file(filename):
    """Reads data from any file for use in orbitize
    readable by ``astropy.io.ascii.read()``, including csv format.
    See the `astropy docs <http://docs.astropy.org/en/stable/io/ascii/index.html#id1>`_.

    There are two ways to provide input data to orbitize.

    The first way is to provide astrometric measurements, shown with the following example.

    Example of an orbitize-readable .csv input file::

        epoch,object,raoff,raoff_err,decoff,decoff_err,radec_corr,sep,sep_err,pa,pa_err,seppa_corr,rv,rv_err,brightness,brightness_err,instrument
        1234,1,0.010,0.005,0.50,0.05,0.25,,,,,,,,,,
        1235,1,,,,,,1.0,0.005,89.0,0.1,,,,0.9,0.1,
        1236,1,,,,,,1.0,0.005,89.3,0.3,-0.5,,,0.6,0.2,
        1237,0,,,,,,,,,,,10.0,0.1,,,A
        1237,0,,,,,,,,,,,13.0,0.2,,,B
        1238,0,,,,,,,,,,,10.2,0.2,,,A
        1238,0,,,,,,,,,,,13.1,0.2,,,B

    Each row must have ``epoch`` (in MJD=JD-2400000.5) and ``object``.
    Objects are numbered with integers, where the primary/central object is ``0``.
    If you have, for example, one RV measurement of a star and three astrometric
    measurements of an orbiting planet, you should put ``0`` in the ``object`` column
    for the RV point, and ``1`` in the columns for the astrometric measurements.

    Each line must also have at least one of the following sets of valid measurements:

        - RA and DEC offsets [mas], or
        - sep [mas] and PA [degrees East of NCP], or
        - RV measurement [km/s], or
        - brightness

    Each line may optionally have an ``instrument`` name. This is particularly important
    if there are multiple RV instruments whose RV offsets need to be fitted separately.

    .. Note:: Columns with no data can be omitted (e.g. if only separation and PA
        are given, the raoff, deoff, and rv columns can be excluded).

        If more than one valid set is given (e.g. RV measurement and astrometric measurement
        taken at the same epoch), ``read_file()`` will generate a separate output row for
        each valid set.

    For RA/Dec and Sep/PA, you can also specify a correlation term. This is useful when your error ellipse
    is tilted with respect to the RA/Dec or Sep/PA. The correlation term is the Pearson correlation coefficient (ρ),
    which corresponds to the normalized off diagonal term of the covariance matrix. Let's define the
    covariance matrix as ``C = [[C_11, C_12], [C_12, C_22]]``. Here C_11 = quant1_err^2 and C_22 = quant2_err^2
    and C_12 is the off diagonal term. Then ``ρ = C_12/(sqrt(C_11)sqrt(C_22))``. Essentially it is the covariance
    normalized by the variance. As such, -1 ≤ ρ ≤ 1. You can specify either as radec_corr or seppa_corr. By definition,
    both are dimensionless, but one will correspond to RA/Dec and the other to Sep/PA. An example of real world data
    that reports correlations is `this GRAVITY paper <https://arxiv.org/abs/2101.04187>`_ where table 2 reports the
    correlation values and figure 4 shows how the error ellipses are tilted.

    Alternatively, you can also supply a data file with the columns already corresponding to
    the orbitize format (see the example in description of what this method returns). This may
    be useful if you are wanting to use the output of the `write_orbitize_input` method.

    .. Note:: RV measurements of objects that are not the primary should be relative to
        the barycenter RV. For example, if the barycenter has a RV
        of 20 +/- 1 km/s, and you've measured an absolute RV for the secondary of 15 +/- 2 km/s,
        you should input an RV of -5.0 +/- 2.2 for object 1.

    .. Note:: When providing data with columns in the orbitize format, there should be no
        empty cells. As in the example below, when quant2 is not applicable, the cell should
        contain nan.

    Args:
        filename (str or astropy.table.Table): Input filename or the actual table object

    Returns:
        astropy.Table: Table containing orbitize-readable input for given
        object. For the example input above::

             epoch  object  quant1 quant1_err  quant2 quant2_err quant12_corr quant_type instrument
            float64 int64  float64  float64   float64  float64     float64     bytes10     str20   
            ------- ------ ------- ---------- ------- ---------- ------------ ---------- ----------
             1234.0      1    0.01      0.005     0.5       0.05         0.25      radec      defrd
             1235.0      1     1.0      0.005    89.0        0.1          nan      seppa      defsp
             1235.0      1     0.9        0.1     nan        nan          nan brightness      defbr
             1236.0      1     1.0      0.005    89.3        0.3         -0.5      seppa      defsp
             1236.0      1     0.6        0.2     nan        nan          nan brightness      defbr
             1237.0      0    10.0        0.1     nan        nan          nan         rv          A
             1237.0      0    13.0        0.2     nan        nan          nan         rv          B
             1238.0      0    10.2        0.2     nan        nan          nan         rv          A
             1238.0      0    13.1        0.2     nan        nan          nan         rv          B

        where ``quant_type`` is one of "radec", "seppa", or "rv".

        If ``quant_type`` is "radec" or "seppa", the units of quant are mas and degrees,
        if ``quant_type`` is "rv", the units of quant are km/s

    Written: Henry Ngo, 2018

    Updated: Vighnesh Nagpal, Jason Wang (2020-2021)
    """
    # initialize output table
    output_table = Table(
        names=(
            "epoch",
            "object",
            "quant1",
            "quant1_err",
            "quant2",
            "quant2_err",
            "quant12_corr",
            "quant_type",
            "instrument",
        ),
        dtype=(float, int, float, float, float, float, float, "S5", "U20"),
    )

    # dict string to array of bools of which rows have each input attribute
    have = {}

    # read file
    try:
        # load from file, unless a table is passed in
        if not isinstance(filename, Table):
            input_table = read(filename)
        else:
            input_table = filename

        # convert to masked table
        if input_table.has_masked_columns:
            input_table = Table(input_table, masked=True, copy=False)

    except:
        raise Exception(
            "Unable to read file: {}. \n Please check file path and format.".format(
                filename
            )
        )
    num_measurements = len(input_table)

    # Decide if input was given in the orbitize style
    orbitize_style = "quant_type" in input_table.columns

    # validate input
    # if input_table is Masked, then figure out which entries are masked
    # otherwise, just check that we have the required columns based on orbitize_style flag
    if input_table.masked:
        if "epoch" in input_table.columns:
            have_epoch = ~input_table["epoch"].mask
            if not have_epoch.all():
                raise Exception("Invalid input format: missing some epoch entries")
        else:
            raise Exception("Input table MUST have epoch!")
        if "object" in input_table.columns:
            have_object = ~input_table["object"].mask
            if not have_object.all():
                raise Exception("Invalid input format: missing some object entries")
        else:
            raise Exception("Input table MUST have object id!")
        if (
            orbitize_style
        ):  # proper orbitize style should NEVER have masked entries (nan required)
            raise Exception("Input table in orbitize style may NOT have empty cells")
        else:  # Check for these things when not orbitize style
            for attr in VALID_ATTRS:
                if attr in input_table.columns:
                    have[attr] = ~input_table[attr].mask
                else:
                    have[attr] = np.zeros(num_measurements, dtype=bool) # zeros are False   
    else:  # no masked entries, just check for required columns
        if "epoch" not in input_table.columns:
            raise Exception("Input table MUST have epoch!")
        if "object" not in input_table.columns:
            raise Exception("Input table MUST have object id!")
        if (
            not orbitize_style
        ):  # Set these flags only when not already in orbitize style
            for attr in VALID_ATTRS:
                if attr in input_table.columns:
                    have[attr] = np.ones(num_measurements, dtype=bool) # ones are True
                else:
                    have[attr] = np.zeros(num_measurements, dtype=bool) # zeros are False

    # orbitize! backwards compatability since we added new columns, some old data formats may not have them
    # fill in with default values
    if orbitize_style:
        if "quant12_corr" not in input_table.keys():
            default_corrs = np.nan * np.ones(len(input_table))
            input_table.add_column(default_corrs, name="quant12_corr")
        if "instrument" not in input_table.keys():
            default_insts = []
            for this_quant_type in input_table["quant_type"]:
                if this_quant_type == "radec":
                    default_insts.append("defrd")
                elif this_quant_type == "seppa":
                    default_insts.append("defsp")
                elif this_quant_type == "rv":
                    default_insts.append("defrv")
                else:
                    raise Exception(
                        "Invalid 'quant_type' {0}. Valid values are 'radec', 'seppa' or 'rv'".format(
                            this_quant_type
                        )
                    )
            input_table.add_column(default_insts, name="instrument")

    # loop through each row and format table
    for index, row in enumerate(input_table):
        # First check if epoch is a number
        try:
            float_epoch = np.float64(row["epoch"])
        except:
            raise Exception(
                "Problem reading epoch in the input file. Epoch should be given in MJD."
            )
        # check epoch format and put in MJD
        if row["epoch"] > 2400000.5:  # assume this is in JD
            print("Converting input epochs from JD to MJD.\n")
            MJD = row["epoch"] - 2400000.5
        else:
            MJD = row["epoch"]

        # check that "object" is an integer (instead of ABC/bcd)
        if not isinstance(row["object"], (int, np.int32, np.int64)):
            raise Exception("Invalid object ID. Object IDs must be integers.")

        # determine input quantity type (RA/DEC, SEP/PA, RV, or BRIGHTNESS)
        if orbitize_style:
            if row["quant_type"] == "rv" or row["quant_type"] == 'brightness':  # special format for rv rows
                output_table.add_row(
                    [
                        MJD,
                        row["object"],
                        row["quant1"],
                        row["quant1_err"],
                        None,
                        None,
                        None,
                        row["quant_type"],
                        row["instrument"],
                    ]
                )

            elif (
                row["quant_type"] == "radec" or row["quant_type"] == "seppa"
            ):  # other allowed
                # backwards compatability whether it has the covariance term or not
                if "quant12_corr" in row.columns:
                    quant12_corr = row["quant12_corr"]
                    if quant12_corr > 1 or quant12_corr < -1:
                        raise ValueError(
                            "Invalid correlation coefficient at line {0}. Value should be between -1 and 1 but got {1}".format(
                                index, quant12_corr
                            )
                        )
                else:
                    quant12_corr = None
                output_table.add_row(
                    [
                        MJD,
                        row["object"],
                        row["quant1"],
                        row["quant1_err"],
                        row["quant2"],
                        row["quant2_err"],
                        quant12_corr,
                        row["quant_type"],
                        row["instrument"],
                    ]
                )
            else:  # catch wrong formats
                raise Exception(
                    "Invalid 'quant_type' {0}. Valid values are 'radec', 'seppa' or 'rv'".format(
                        row["quant_type"]
                    )
                )

        else:  # When not in orbitize style

            if have["raoff"][index] and have["decoff"][index]:
                # check if there's a covariance term
                if have["radec_corr"][index]:
                    quant12_corr = row["radec_corr"]
                    if quant12_corr > 1 or quant12_corr < -1:
                        raise ValueError(
                            "Invalid correlation coefficient at line {0}. Value should be between -1 and 1 but got {1}".format(
                                index, quant12_corr
                            )
                        )
                else:
                    quant12_corr = None

                if have["instrument"][index]:
                    this_inst = row["instrument"]
                else:
                    # sets the row with a default instrument name if none is provided
                    this_inst = "defrd"

                output_table.add_row(
                    [
                        MJD,
                        row["object"],
                        row["raoff"],
                        row["raoff_err"],
                        row["decoff"],
                        row["decoff_err"],
                        quant12_corr,
                        "radec",
                        this_inst,
                    ]
                )

            elif have["sep"][index] and have["pa"][index]:
                # check if there's a covariance term
                if have["seppa_corr"][index]:
                    quant12_corr = row["seppa_corr"]
                    if quant12_corr > 1 or quant12_corr < -1:
                        raise ValueError(
                            "Invalid correlation coefficient at line {0}. Value should be between -1 and 1 but got {1}".format(
                                index, quant12_corr
                            )
                        )
                else:
                    quant12_corr = None

                if have["instrument"][index]:
                    this_inst = row["instrument"]
                else:
                    # sets the row with a default instrument name if none is provided
                    this_inst = "defsp"

                output_table.add_row(
                    [
                        MJD,
                        row["object"],
                        row["sep"],
                        row["sep_err"],
                        row["pa"],
                        row["pa_err"],
                        quant12_corr,
                        "seppa",
                        this_inst,
                    ]
                )

            if have["rv"][index]:
                if have["instrument"][index]:
                    output_table.add_row(
                        [
                            MJD,
                            row["object"],
                            row["rv"],
                            row["rv_err"],
                            None,
                            None,
                            None,
                            "rv",
                            row["instrument"],
                        ]
                    )
                else:
                    # Vighnesh: sets the row with a default instrument name if none is provided
                    output_table.add_row(
                        [
                            MJD,
                            row["object"],
                            row["rv"],
                            row["rv_err"],
                            None,
                            None,
                            None,
                            "rv",
                            "defrv",
                        ]
                    )
            if have["brightness"][index]:
                output_table.add_row(
                    [
                        MJD,
                        row["object"],
                        row["brightness"],
                        row["brightness_err"],
                        None,
                        None,
                        None,
                        "brightness",
                        "defbr",
                    ]
                    )

    return output_table


def write_orbitize_input(table, output_filename, file_type="csv"):
    """Writes orbitize-readable input as an ASCII file

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
    valid_formats = [
        "aastex",
        "basic",
        "commented_header",
        "csv",
        "ecsv",
        "fixed_width",
        "fixed_width_no_header",
        "fixed_width_two_line",
        "html",
        "ipac",
        "latex",
        "no_header",
        "rdb",
        "rst",
        "tab",
    ]
    if file_type not in valid_formats:
        raise Exception("Invalid output format specified.")

    # write file
    write(table, output=output_filename, format=file_type)
