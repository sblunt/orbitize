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

If you use ``orbitize`` in your work, please cite our forthcoming
paper and the following DOI: 

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3337378.svg
   :target: https://zenodo.org/record/3337378#.XUHT3ZNKjUJ

If you use the OFTI algorithm, please also cite `Blunt et al (2017) <https://ui.adsabs.harvard.edu/#abs/2017AJ....153..229B/abstract>`_.

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

**Some major planned updates:**

- fit orbits of multiple objects in one system
- fit Gaia astrometry & RVs
- marginalize over instrumental uncertainties

Changelog:
++++++++++

**1.6.0 (2019-10-1)**

- Mikkola solver now implemented in C-Kepler solver
- Fixed a bug with parallel processing for OFTI
- Added orbit vizualisation jupyter nb show-me-the-orbit to docs/tutorials
- New methods for viewing/chopping MCMC chains
- Require `emcee<3` for now

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
