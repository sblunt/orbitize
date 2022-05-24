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

* If you use ``orbitize`` in your work, please cite `Blunt et al (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv191001756B/abstract>`_.
* If you use the OFTI algorithm, please also cite `Blunt et al (2017) <https://ui.adsabs.harvard.edu/#abs/2017AJ....153..229B/abstract>`_. 
* If you use the Affine-invariant MCMC algorithm from ``emcee``, please also cite `Foreman-Mackey et al (2013) <https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract>`_. 
* If you use the parallel-tempered Affine-invariant MCMC algorithm from ``ptemcee``, please also cite `Vousden et al (2016) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.1919V/abstract>`_. 
* If you use the Hipparcos intermediate astrometric data (IAD) fitting capability, please also cite `Nielsen et al (2020) <https://ui.adsabs.harvard.edu/abs/2020AJ....159...71N/abstract>`_ and `van Leeuwen et al (2007)  <https://ui.adsabs.harvard.edu/abs/2007A%26A...474..653V/abstract>`_.
* If you use Gaia data, please also cite `Gaia Collaboration et al (2018; for DR2) <https://ui.adsabs.harvard.edu/abs/2018A%26A...616A...1G/abstract>`_, or `Gaia Collaboration et al (2021; for eDR3) <https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...1G/abstract>`_.


User Guide:
+++++++++++

.. toctree::
   :maxdepth: 2

   installation
   tutorials
   faq
   contributing
   api

Changelog:
++++++++++

**2.1.0 (2022-05-24)**

- Added a (more numerically stable) log-chi2 option for calculating likelihood (@Mireya-A and @lhirsch238)

**2.0.1 (2022-04-22)**

- Addressed plotting bugs: issues #316/#309, #314, #311 (@semaphoreP)
- Made Gaia module runnable without internet and added some Gaia/Hipparcos unit tests (@sblunt) 

**2.0.0 (2021-10-13)**

This is the official release of orbitize! version 2.

Big changes:

- Fit Gaia positions (@sblunt)
- New plotting module & API (@sblunt)
- Relative planet RVs now officially supported & tested (@sblunt)
- GPU Kepler solver (@devincody)
- RV end-to-end test added (@vighnesh-nagpal)

Small changes:

- Hipparcos calculation bugfix (@sblunt)
- v1 results backwards compatibility bugfix (@sblunt)
- windows install docs update (@sblunt
- basis bugfix with new API (@TirthDS, @sblunt)
- handle Hipparcos 2021 data format (@sblunt)
- clarify API on mtot/mstar (@lhirsch238, @sblunt)

**2.0b1 (2021-09-03)**

This is the beta release of orbitize! version 2.

Big changes:

- N-body Kepler solver backend! (@sofiacovarrubias)
- Fitting in XYZ orbital basis! (@rferrerc)
- API for fitting in arbitrary new orbital bases! (@TirthDS)
- compute_all_orbits separated out, streamlining stellar astrometry & RV calculations (@sblunt)
- Hip IAD! (@sblunt)
- param_idx now used everywhere under the hood (system parsing updated) (@sblunt)
- KDE prior added (inspiration=training on RV fits) (@jorgellop)

Small changes:

- HD 4747 rv data file fix for the RV tutorial (@lhirsch238)
- Add check_prior_support to sampler.MCMC (@adj-smith)
- Update example generation code in MCMC v OFTI tutorial (@semaphoreP)
- Fixed plotting bug (issue #243) (@TirthDS)
- Expand FAQ section (@semaphoreP)
- use astropy tables in results (@semaphoreP)
- Expand converge section of MCMC tutorial (@semaphoreP)
- Deprecated functions and deprecation warnings officially removed (@semaphoreP)
- Fix logic in setting of track_planet_perturbs (@sblunt)
- Fix plotting error if orbital periods are > 1e9 days (@sblunt)
- Add method for printing results of a fit (@sblunt)

**1.16.1 (2021-06-27)**

* Fixed chop_chains() function to copy original data over when updating Results object (@TirthDS)

**1.16.0 (2021-06-23)**

* User-defined prior on PAN were not being applied if OFTI is used; fixed (@sblunt)
* Dates in HD 4747 data file were incorrect; fixed (@lhirsch238)

**1.15.5 (2021-07-20)**

* Addressed issue #177, giving `Results` and `Sampler` classes a parameter label array (@sblunt)
* Fixed a bug that was causing RA/Dec data points to display wrong in orbit plots (@sblunt)

**1.15.4 (2021-06-18)**

* Bugfix for issue #234 (@semaphoreP, @adj-smith)

**1.15.3 (2021-06-07)**

* Add codeastro mode to pytest that prints out a SECRET CODE if tests pass omgomg (@semaphoreP)

**1.15.2 (2021-05-11)**

* Fixed backwards-compatibility bug with version numbers and saving/loading (@semaphoreP, @wbalmer)

**1.15.1 (2021-03-29)**

* Fixed bug where users with Results objects from v<14.0 couldn't load using v>=14.0 (@semaphoreP, @wbalmer)
* Fixed order of Axes objects in Advanced Plotting tutorial (@wbalmer, @sblunt)

**1.15.0 (2021-02-23)**

* Handle covariances in input astrometry (@semaphoreP)

**1.14.0 (2021-02-12)**

* Version number now saved in results object (@hgallamore)
* Joint RV+astrometry fits can now handle different RV instruments! (@vighnesh-nagpal, @Rob685, @lhirsch238)
* New “FAQ” section added to docs (@semaphoreP)
* Bugfix for multiplanet code (@semaphoreP) introduced in PR #192 
* now you can pass a preexisting Figure object into ``results.plot_orbit`` (@sblunt)
* colorbar label is now "Epoch [year]" (@sblunt)
* corner plot maker can now handle fixed parameters without crashing (@sblunt)

**1.13.1 (2021-01-25)**

* ``compute_sep`` in ``radvel_utils`` submodule now returns ``mp`` (@sblunt)
* ``astropy._erfa`` was deprecated (now in separate package). Dependencies updated. (@sblunt)

**1.13.0 (2020-11-8)**

* Added ``radvel-utils`` submodule which allows users to calculate projected separation posteriors given RadVel chains (@sblunt)
* Fixed a total mass/primary mass mixup bug that was causing problems for equal-mass binary RV+astrometry joint fits (@sblunt)
* Bugfix for multiplanet perturbation approximation: now only account for inner planets only when computing perturbations (@semaphoreP)

**1.12.1 (2020-9-6)**

* ``tau_ref_epoch`` is now set to Jan 1, 2020 throughout the code (@semaphoreP)
* ``restrict_angle_ranges`` keyword now works as expected for OFTI (@sblunt)

**1.12.0 (2020-8-28)**

* Compatibility with ``emcee>=3`` (@sblunt)

**1.11.3 (2020-8-20)**

* Save results section of OFTI tutorial now current (@rferrerc)
* Modifying MCMC initial positions tutorial documentation now uses correct orbital elements (@rferrerc)

**1.11.2 (2020-8-10)**

* Added transparency option for plotting MCMC chains (@sofiacovarrubias)
* Removed some redundant code (@MissingBrainException)

**1.11.1 (2020-6-11)**

* Fixed a string formatting bug causing corner plots to fail for RV+astrometry fits

**1.11.0 (2020-4-14)**

* Multiplanet support!
* Changes to directory structure of sample data files
* Fixed a bug that was causing corner plots to fail on loaded results objects

**1.10.0 (2020-3-6)**

* Joint RV + relative astrometry fitting capabilities! 
* New tutorial added

**1.9.0 (2020-1-24)**

* Require astropy>=4
* Minor documentation upgrades
* **This is the first Python 2 noncompliant version**

**1.8.0 (2020-1-24)**

* Bugfixes related to numpy and astropy upgrades
* **This is the last version that will support Python 2**

**1.7.0 (2019-11-10)**

* Default corner plots now display angles in degrees instead of radians
* Add a keyword for plotting orbits that cross PA=360

**1.6.0 (2019-10-1)**

* Mikkola solver now implemented in C-Kepler solver
* Fixed a bug with parallel processing for OFTI
* Added orbit vizualisation jupyter nb show-me-the-orbit to docs/tutorials
* New methods for viewing/chopping MCMC chains
* Require ``emcee<3`` for now

**1.5.0 (2019-9-9)**

* Parallel processing for OFTI.
* Fixed a bug converting errors in RA/Dec to sep/PA in OFTI.
* OFTI and MCMC now both return likelihood, whereas before one returned posterior.
* Updated logic for restricting Omega and omega bounds.

**1.4.0 (2019-7-15)**

* API change to lay the groundwork for dynamical mass calculation. 
* JeffreysPrior -> LogUniformPrior
* New tutorials.
* Added some informative error messages for input tables.
* Bugfixes.


**1.3.1 (2019-6-19)**

* Bugfix for RA/Dec inputs to the OFTI sampler (Issue #108).

**1.3.0 (2019-6-4)**

* Add ability to customize date of tau definition. 
* Sampler now saves choice of tau reference with results.
* Default tau value is now Jan 1, 2020.
* Small bugfixes.

**1.2.0 (2019-3-21)**

* Remove unnecessary ``astropy`` date warnings.
* Add custom likelihood function.
* Add progress bar for ``ptemcee`` sampler.
* Add customizable color axis for orbit plots.
* Small bugfixes.

**1.1.0 (2019-1-6)**

* Add sep/PA panels to orbit plot.
* ``GaussianPrior`` now operates on only positive numbers by default.

**1.0.2 (2018-12-4)**

* Expand input reading functionality.
* Bugfixes for MCMC.

**1.0.1 (2018-11-20)**

* Bugfix for building on CentOS machines.

**1.0.0 (2018-10-30)**

* Initial release.
