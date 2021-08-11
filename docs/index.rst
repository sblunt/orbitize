.. |br| raw:: html

   <br />

.. image:: orbitize_logo_500.png
   :width: 150px
   :height: 150px
   :align: center


orbitize!
=========

Hello world! Welcome to the documentation for ``orbitize``, a Python
package for fitting orbits of directly imaged planets. 

``orbitize`` packages two back-end algorithms into a consistent API. 
It's written to be fast, extensible, and easy-to-use. The tutorials below will walk
you through the code and introduce some technical stuff, but we suggest learning about
the `Orbits for the Impatient (OFTI) algorithm <https://ui.adsabs.harvard.edu/#abs/2017AJ....153..229B/abstract>`_
and MCMC algorithms (we use `this one <http://dfm.io/emcee/current/>`_) before diving in.
Our `contributor guidelines <https://github.com/sblunt/orbitize/blob/master/contributor_guidelines.md>`_
document will point you to more useful resources. 

``orbitize`` is designed to meet the needs of the exoplanet imaging community, and we
encourage community involvement. If you find a bug, want to request a feature, etc. please
create an `issue on GitHub <https://github.com/sblunt/orbitize/issues>`_. 

``orbitize`` is patterned after and inspired by `radvel <https://radvel.readthedocs.io/en/latest/>`_. 

Attribution:
++++++++++++

- If you use ``orbitize`` in your work, please cite `Blunt et al (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv191001756B/abstract>`_.
- If you use the OFTI algorithm, please also cite `Blunt et al (2017) <https://ui.adsabs.harvard.edu/#abs/2017AJ....153..229B/abstract>`_. 
- If you use the Affine-invariant MCMC algorithm from ``emcee``, please also cite `Foreman-Mackey et al (2013) <https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract>`_. 
- If you use the parallel-tempered Affine-invariant MCMC algorithm from ``ptemcee``, please also cite `Vousden et al (2016) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1919V/abstract>`_. 



User Guide:
+++++++++++

.. toctree::
   :maxdepth: 2

   installation
   tutorials
   api

Contributing:
+++++++++++++

``orbitize`` is under active development, and we've still got a lot to do! To get involved,
check out our `contributor guidelines <https://github.com/sblunt/orbitize/blob/master/contributor_guidelines.md>`_,
look over our `issues list <https://github.com/sblunt/orbitize/issues>`_, and/or reach out to 
`Sarah <https://sites.google.com/g.harvard.edu/sarah/contact?authuser=0>`_. We'd love to have
you on our team!

Members of our team have collectively drafted `this community agreement <https://docs.google.com/document/d/1ZzjkoB20vVTlg2wbNpS7sRjmcSrECdh8kQ11-waZQhw/edit>`_ stating both our values and ground rules. 
In joining our team, we ask that you read and (optionally) suggest changes to this document. 


**Some major planned updates:**

- fit Gaia astrometry
- marginalize over instrumental uncertainties

Changelog:
++++++++++

**1.13.1 (2020-01-25)**

- ``compute_sep`` in ``radvel_utils`` submodule now returns ``mp`` (@sblunt)
- ``astropy._erfa`` was deprecated (now in separate package). Dependencies updated. (@sblunt)

**1.13.0 (2020-11-8)**

- Added ``radvel-utils`` submodule which allows users to calculate projected separation posteriors given RadVel chains (@sblunt)
- Fixed a total mass/primary mass mixup bug that was causing problems for equal-mass binary RV+astrometry joint fits (@sblunt)
- Bugfix for multiplanet perturbation approximation: now only account for inner planets only when computing perturbations (@semaphoreP)

**1.12.1 (2020-9-6)**

- ``tau_ref_epoch`` is now set to Jan 1, 2020 throughout the code (@semaphoreP)
- ``restrict_angle_ranges`` keyword now works as expected for OFTI (@sblunt)

**1.12.0 (2020-8-28)**

- Compatibility with ``emcee>=3`` (@sblunt)

**1.11.3 (2020-8-20)**

- Save results section of OFTI tutorial now current (@rferrerc)
- Modifying MCMC initial positions tutorial documentation now uses correct orbital elements (@rferrerc)

**1.11.2 (2020-8-10)**

- Added transparency option for plotting MCMC chains (@sofiacovarrubias)
- Removed some redundant code (@MissingBrainException)

**1.11.1 (2020-6-11)**

- Fixed a string formatting bug causing corner plots to fail for RV+astrometry fits

**1.11.0 (2020-4-14)**

- Multiplanet support!
- Changes to directory structure of sample data files
- Fixed a bug that was causing corner plots to fail on loaded results objects

**1.10.0 (2020-3-6)**

- Joint RV + relative astrometry fitting capabilities! 
- New tutorial added

**1.9.0 (2020-1-24)**

- Require astropy>=4
- Minor documentation upgrades
- **This is the first Python 2 noncompliant version**

**1.8.0 (2020-1-24)**

- Bugfixes related to numpy and astropy upgrades
- **This is the last version that will support Python 2**

**1.7.0 (2019-11-10)**

- Default corner plots now display angles in degrees instead of radians
- Add a keyword for plotting orbits that cross PA=360

**1.6.0 (2019-10-1)**

- Mikkola solver now implemented in C-Kepler solver
- Fixed a bug with parallel processing for OFTI
- Added orbit vizualisation jupyter nb show-me-the-orbit to docs/tutorials
- New methods for viewing/chopping MCMC chains
- Require ``emcee<3`` for now

**1.5.0 (2019-9-9)**

- Parallel processing for OFTI.
- Fixed a bug converting errors in RA/Dec to sep/PA in OFTI.
- OFTI and MCMC now both return likelihood, whereas before one returned posterior.
- Updated logic for restricting Omega and omega bounds.

**1.4.0 (2019-7-15)**

- API change to lay the groundwork for dynamical mass calculation. 
- JeffreysPrior -> LogUniformPrior
- New tutorials.
- Added some informative error messages for input tables.
- Bugfixes.


**1.3.1 (2019-6-19)**

- Bugfix for RA/Dec inputs to the OFTI sampler (Issue #108).

**1.3.0 (2019-6-4)**

- Add ability to customize date of tau definition. 
- Sampler now saves choice of tau reference with results.
- Default tau value is now Jan 1, 2020.
- Small bugfixes.

**1.2.0 (2019-3-21)**

- Remove unnecessary ``astropy`` date warnings.
- Add custom likelihood function.
- Add progress bar for ``ptemcee`` sampler.
- Add customizable color axis for orbit plots.
- Small bugfixes.

**1.1.0 (2019-1-6)**

- Add sep/PA panels to orbit plot.
- ``GaussianPrior`` now operates on only positive numbers by default.


**1.0.2 (2018-12-4)**

- Expand input reading functionality.
- Bugfixes for MCMC.

**1.0.1 (2018-11-20)**

- Bugfix for building on CentOS machines.

**1.0.0 (2018-10-30)**

- Initial release.
