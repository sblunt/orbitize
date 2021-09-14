.. _formatting_inputs:

Formatting Input
++++++++++++++++

Use ``orbitize.read_input.read_file()`` to read your astrometric data into orbitize. This method takes one argument, a string to the path of the file containing your input.

This method can read any file format supported by ``astropy.io.ascii.read()``, including csv format. See the `astropy docs <http://docs.astropy.org/en/stable/io/ascii/index.html#id1>`_.

There are two ways to provide input data to orbitize, either as observations or as an ``orbitize!``-formatted input table. 

Option 1
--------
You can provide your observations in one of the following valid sets of measurements using the corresponding column names: 

    - RA and DEC offsets [milliarcseconds],  using column names ``raoff``, ``raoff_err``, ``decoff``, and ``decoff_err``; or
    - sep [milliarcseconds] and PA [degrees East of NCP], using column names ``sep``, ``sep_err``, ``pa``, and ``pa_err``; or
    - RV measurement [km/s] using column names ``rv`` and ``rv_err``.

Each row must also have a column for ``epoch`` and ``object``. Epoch is the date of the observation, in MJD (JD-2400000.5). If this method thinks you have provided a date in JD, it will print a warning and attempt to convert to MJD. Objects are numbered with integers, where the primary/central object is ``0``.

You may mix and match these three valid measurement formats in the same input file. So, you can have some epochs with RA/DEC offsets and others in separation/PA measurements.

If you have, for example, one RV measurement of a star and three astrometric
measurements of an orbiting planet, you should put ``0`` in the ``object`` column for the RV point, and ``1`` in the columns for the astrometric measurements.

This method will look for columns with the above labels in whatever file format you choose so if you encounter errors, be sure to double check the column labels in your input file.

Putting it all together, here an example of a valid .csv input file::

    epoch,object,raoff,raoff_err,decoff,decoff_err,radec_corr,sep,sep_err,pa,pa_err,rv,rv_err
    1234,1,0.010,0.005,0.50,0.05,0.025,,,,,,
    1235,1,,,,,,1.0,0.005,89.0,0.1,,
    1236,1,,,,,,1.0,0.005,89.3,0.3,,
    1237,0,,,,,,,,,,10,0.1

.. Note:: Columns with no data can be omitted (e.g. if only separation and PA
    are given, the raoff, deoff, and rv columns can be excluded).

    If more than one valid set is given (e.g. RV measurement and astrometric measurement taken at the same epoch), ``read_file()`` will generate a separate output row for each valid set.

Whatever file format you choose, this method will read your input into an ``orbitize!``-formatted input table. This is an ``astropy.Table.table`` object that looks like this (for the example input given above)::

        epoch  object  quant1 quant1_err  quant2 quant2_err quant12_corr quant_type
        float64  int   float64  float64   float64  float64     float64       str5
        ------- ------ ------- ---------- ------- ---------- ------------ ----------
        1234.0      1    0.01      0.005     0.5       0.05      0.025        radec
        1235.0      1     1.0      0.005    89.0        0.1        nan        seppa
        1236.0      1     1.0      0.005    89.3        0.3        nan        seppa
        1237.0      0    10.0        0.1     nan        nan        nan           rv

where ``quant_type`` is one of "radec", "seppa", or "rv".

If ``quant_type`` is "radec" or "seppa", the units of quant are mas and degrees,
if ``quant_type`` is "rv", the units of quant are km/s.

Covariances
^^^^^^^^^^^
For RA/Dec and Sep/PA, you can optionally specify a correlation term. This is useful when your error ellipse
is tilted with respect to the RA/Dec or Sep/PA. The correlation term is the Pearson correlation coefficient (ρ),
which corresponds to the normalized off diagonal term of the covariance matrix (C):

.. math::

   C = \begin{bmatrix}
    C_{11} & C_{12}\\
    C_{12} & C_{22}
    \end{bmatrix}.

Here C_11 = quant1_err^2 and C_22 = quant2_err^2
and C_12 is the off diagonal term (note that by definition both off-diagonal terms of the covariance matrix are the same). 
Then, :math:`\rho = C_{12}/\sqrt{C_{11}C_{22}}`. Essentially it is the covariance 
normalized by the variance. As such, -1 ≤ ρ ≤ 1. 
You can specify either as radec_corr or seppa_corr to include a correlation in the errors. By definition,
both are dimensionless, but one will correspond to RA/Dec and the other to Sep/PA. If no correlations are specified, it will assume
the errors are uncorrelated (ρ = 0). In many papers, the errors are assumed to be uncorrelated. An example of real world data
that reports correlations is `this GRAVITY paper <https://arxiv.org/abs/2101.04187>`_ where table 2 reports the
correlation values and figure 4 shows how the error ellipses are tilted. 

In the example above, we specify the first epoch has a positive correlation between the uncertainties in RA and Dec using the
``radec_corr`` column in the input data. This gets translated into the ``quant12_corr`` field in ``orbitize!``-format. No
correlations are specified for the other entries, and so we will assume those errors are uncorrelated. 
After this is specified, handling of the correlations will be done automatically when computing model likelihoods.
There's nothing else you have to do after this step!


Option 2
--------
Alternatively, you can also supply a data file with the columns already corresponding to the ``orbitize!``-formatted input table (see above for column names). This may be useful if you are wanting to use the output of the ``write_orbitize_input`` method (e.g. using some input prepared by another ``orbitize!`` user).

.. Note:: When providing data with columns in the orbitize format, there should be
    no empty cells. As in the example below, when quant2 is not applicable, the cell should contain nan.
