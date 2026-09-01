"""
Module to read user input from files and create standardized input for orbitize
"""

import numpy as np
import orbitize
from astropy.table import Table
from astropy.io.ascii import read, write
from astropy import units as u


def read_file(filename):
    """Reads data from any file for use in orbitize
    readable by ``astropy.io.ascii.read()``, including csv format.
    See the `astropy docs <http://docs.astropy.org/en/stable/io/ascii/index.html#id1>`_.

    There are two ways to provide input data to orbitize.

    The first way is to provide astrometric measurements, shown with the following example.

    Example of an orbitize-readable .csv input file::

        epoch,object,raoff,raoff_err,decoff,decoff_err,radec_corr,sep,sep_err,pa,pa_err,rv,rv_err
        1234,1,0.010,0.005,0.50,0.05,0.025,,,,,,
        1235,1,,,,,,1.0,0.005,89.0,0.1,,
        1236,1,,,,,,1.0,0.005,89.3,0.3,,
        1237,0,,,,,,,,,,10,0.1

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

            epoch  object  quant1 quant1_err  quant2 quant2_err quant12_corr quant_type
           float64  int   float64  float64   float64  float64     float64       str5
           ------- ------ ------- ---------- ------- ---------- ------------ ----------
           1234.0      1    0.01      0.005     0.5       0.05      0.025        radec
           1235.0      1     1.0      0.005    89.0        0.1        nan        seppa
           1236.0      1     1.0      0.005    89.3        0.3        nan        seppa
           1237.0      0    10.0        0.1     nan        nan        nan           rv

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
            if "raoff" in input_table.columns:
                have_ra = ~input_table["raoff"].mask
            else:
                have_ra = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if "decoff" in input_table.columns:
                have_dec = ~input_table["decoff"].mask
            else:
                have_dec = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if "radec_corr" in input_table.columns:
                have_radeccorr = ~input_table["radec_corr"].mask
            else:
                have_radeccorr = np.zeros(
                    num_measurements, dtype=bool
                )  # zeros are False
            if "sep" in input_table.columns:
                have_sep = ~input_table["sep"].mask
            else:
                have_sep = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if "pa" in input_table.columns:
                have_pa = ~input_table["pa"].mask
            else:
                have_pa = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if "seppa_corr" in input_table.columns:
                have_seppacorr = ~input_table["seppa_corr"].mask
            else:
                have_seppacorr = np.zeros(
                    num_measurements, dtype=bool
                )  # zeros are False
            if "rv" in input_table.columns:
                have_rv = ~input_table["rv"].mask
            else:
                have_rv = np.zeros(num_measurements, dtype=bool)  # zeros are False

            if "instrument" in input_table.columns:
                # Vighnesh: establishes which rows have instrument names provided
                have_inst = ~input_table["instrument"].mask
            else:
                have_inst = np.zeros(num_measurements, dtype=bool)  # zeros are false

    else:  # no masked entries, just check for required columns
        if "epoch" not in input_table.columns:
            raise Exception("Input table MUST have epoch!")
        if "object" not in input_table.columns:
            raise Exception("Input table MUST have object id!")
        if (
            not orbitize_style
        ):  # Set these flags only when not already in orbitize style
            if "raoff" in input_table.columns:
                have_ra = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_ra = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if "decoff" in input_table.columns:
                have_dec = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_dec = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if "radec_corr" in input_table.columns:
                have_radeccorr = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_radeccorr = np.zeros(
                    num_measurements, dtype=bool
                )  # zeros are False
            if "sep" in input_table.columns:
                have_sep = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_sep = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if "pa" in input_table.columns:
                have_pa = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_pa = np.zeros(num_measurements, dtype=bool)  # zeros are False
            if "seppa_corr" in input_table.columns:
                have_seppacorr = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_seppacorr = np.zeros(
                    num_measurements, dtype=bool
                )  # zeros are False
            if "rv" in input_table.columns:
                have_rv = np.ones(num_measurements, dtype=bool)  # ones are False
            else:
                have_rv = np.zeros(num_measurements, dtype=bool)  # zeros are False

            # Rob: not sure if we need this but adding just in case
            if "instrument" in input_table.columns:
                have_inst = np.ones(num_measurements, dtype=bool)
            else:
                have_inst = np.zeros(num_measurements, dtype=bool)

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

        # determine input quantity type (RA/DEC, SEP/PA, or RV)
        if orbitize_style:
            if row["quant_type"] == "rv":  # special format for rv rows
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
            if have_ra[index] and have_dec[index]:
                # check if there's a covariance term
                if have_radeccorr[index]:
                    quant12_corr = row["radec_corr"]
                    if quant12_corr > 1 or quant12_corr < -1:
                        raise ValueError(
                            "Invalid correlation coefficient at line {0}. Value should be between -1 and 1 but got {1}".format(
                                index, quant12_corr
                            )
                        )
                else:
                    quant12_corr = None

                if have_inst[index]:
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

            elif have_sep[index] and have_pa[index]:
                # check if there's a covariance term
                if have_seppacorr[index]:
                    quant12_corr = row["seppa_corr"]
                    if quant12_corr > 1 or quant12_corr < -1:
                        raise ValueError(
                            "Invalid correlation coefficient at line {0}. Value should be between -1 and 1 but got {1}".format(
                                index, quant12_corr
                            )
                        )
                else:
                    quant12_corr = None

                if have_inst[index]:
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

            if have_rv[index]:
                if have_inst[index]:
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

    return output_table


def from_ohp_file(filename):
    """Read flat OHP-format Gaia epoch astrometry, including mock data.

    Each row is one observation with these required scalar columns:

        - ``obs_time_tcb``: Julian date in TCB (days, not MJD or nanoseconds).
        - ``centroid_pos_al``: along-scan position in mas.
        - ``centroid_pos_error_al``: positive along-scan uncertainty in mas.
        - ``parallax_factor_al``: dimensionless along-scan parallax factor.
        - ``scan_pos_angle``: scan position angle in radians.

    Input JD times are converted to MJD(TCB). Tables already marked with
    ``meta['time_format']='mjd'`` are not converted again. This reader does not
    apply a barycentric correction, rescale uncertainties, average observations,
    or sort/drop rows. Native Gaia EAS arrays require a separate reader.
    Pass normalized tables directly, or save as ECSV to preserve time metadata;
    a plain CSV does not retain the JD/MJD label.

    Args:
        filename (str, pathlib.Path, or astropy.table.Table): OHP file readable
            by ``astropy.io.ascii.read`` (e.g. CSV), or an existing table.

    Returns:
        astropy.table.Table: Independent table with the five required columns
        stored as float64 values with units. Additional columns, including
        optional numeric or text ``field_of_view``, are preserved. Metadata
        records ``time_format='mjd'``, ``time_scale='tcb'``, and
        ``input_format='ohp'``. This is a Gaia-specific table, not the
        RV/relative-astrometry table returned by :func:`read_file`.

    Raises:
        ValueError: If the table is empty, required columns are missing, or
            required values are masked, nonfinite, nonscalar, nonnumeric,
            have conflicting units, or include nonpositive uncertainties.

    Written: Clarissa Do O, 2026
    """
    output_table = filename.copy() if isinstance(filename, Table) else read(filename)
    time_format = output_table.meta.get("time_format", "jd")
    if time_format not in ("jd", "mjd") or output_table.meta.get("time_scale", "tcb") != "tcb":
        raise ValueError("Gaia epoch times must be JD or MJD in the TCB time scale.")
    #check all needed data
    required_units = {
        "obs_time_tcb": u.day,
        "centroid_pos_al": u.mas,
        "centroid_pos_error_al": u.mas,
        "parallax_factor_al": u.dimensionless_unscaled,
        "scan_pos_angle": u.rad,
    }

    missing = [name for name in required_units if name not in output_table.colnames]
    if missing:
        raise ValueError("Missing required OHP columns: {}".format(", ".join(missing)))
    if len(output_table) == 0:
        raise ValueError("OHP input must contain at least one observation.")

    for name, unit in required_units.items():
        column = output_table[name]
        if column.ndim != 1 or np.iscomplexobj(column):
            raise ValueError(
                "OHP column '{}' must contain real scalar values.".format(name)
            )
        # Inspect the mask before converting: np.asarray discards Astropy masks.
        if np.any(np.ma.getmaskarray(column)):
            raise ValueError("OHP column '{}' contains missing values.".format(name))
        input_unit = getattr(column, "unit", None)
        if input_unit is not None and input_unit != unit:
            raise ValueError(
                "OHP column '{}' must use units of {}.".format(
                    name, str(unit) or "dimensionless"
                )
            )
        try:
            values = np.asarray(column, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "OHP column '{}' must contain numeric scalars.".format(name)
            ) from exc
        if not np.all(np.isfinite(values)):
            raise ValueError("OHP column '{}' contains nonfinite values.".format(name))
        output_table[name] = values
        output_table[name].unit = unit

    if np.any(output_table["centroid_pos_error_al"] <= 0):
        raise ValueError("OHP column 'centroid_pos_error_al' must be strictly positive.")

    if time_format == "jd":
        if np.any(output_table["obs_time_tcb"] < 2400000.5):
            raise ValueError("Expected raw OHP JD times; mark MJD tables with meta['time_format']='mjd'.")
        output_table["obs_time_tcb"] -= 2400000.5
    output_table.meta.update(time_format="mjd", time_scale="tcb")
    output_table.meta.setdefault("input_format", "ohp")
    return output_table


def from_dl2(filename, source_id, *, agis_only=True):
    """Read native EAS prerelease XML (2026-06-26) for one star.

    Specify the star's integer Gaia ``source_id``.
    Returns one row per CCD, keeping AGIS-used measurements by default.
    No additional validity filtering or transit averaging.

    The five measurement columns follow :func:`from_ohp_file`: barycentrically
    corrected MJD(TCB), AL position/error in mas, parallax factor, and scan angle
    in radians. Native times/corrections must be in ns and angles in degrees.
    Source/transit IDs, CCD index (0=SM, 1..9=AF), AGIS flag, field of view,
    and the position reference ``ra0, dec0`` are retained; other fields are not.
    """
    data = Table.read(filename, format="votable")
    data = data[data["source_id"] == source_id]

    # Unpack each CCD into a row; repeat the shared transit information.
    sizes = [np.size(value) for value in data["obs_time_tcb"]]
    output_table = Table()
    for name in (
        "obs_time_tcb", "centroid_pos_al", "centroid_pos_error_al",
        "scan_pos_angle", "used_by_agis_al",
    ):
        if [np.shape(value) for value in data[name]] != [(size,) for size in sizes]:
            raise ValueError("EAS column '{}' must have one aligned CCD array per transit.".format(name))
        output_table[name] = np.ma.concatenate(data[name])
    transit_rows = np.repeat(np.arange(len(data)), sizes)
    for name in ("source_id", "transit_id", "ra0", "dec0", "parallax_factor_al"):
        output_table[name] = data[name][transit_rows]
    output_table["component_index"] = np.concatenate([np.arange(size) for size in sizes])

    # ns (Gaia) -> barycentric MJD; degrees -> radians.
    time_ns = np.ma.asarray(output_table["obs_time_tcb"], dtype=np.float64) # ns
    correction_ns = np.ma.asarray(data["obs_time_bary_corr"][transit_rows], dtype=np.float64) # barycentric corr
    output_table["obs_time_tcb"] = 55197.0 + (time_ns + correction_ns) / 86400e9
    output_table["scan_pos_angle"] = np.deg2rad(
        np.ma.asarray(output_table["scan_pos_angle"], dtype=np.float64)
    )

    # Keep the measurements Gaia used in AGIS.
    if agis_only:
        output_table = output_table[np.ma.filled(output_table["used_by_agis_al"], False)]

    for name, unit in {
        "obs_time_tcb": u.day,
        "centroid_pos_al": u.mas,
        "centroid_pos_error_al": u.mas,
        "scan_pos_angle": u.rad,
        "parallax_factor_al": u.dimensionless_unscaled,
        "ra0": u.deg,
        "dec0": u.deg,
    }.items():
        output_table[name].unit = unit
    output_table["field_of_view"] = 1 + ((np.asarray(output_table["transit_id"]) >> 15) & 0x03)
    output_table.meta.update(
        time_format="mjd", time_scale="tcb", input_format="eas_prerelease",
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
