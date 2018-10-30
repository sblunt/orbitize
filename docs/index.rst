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

Some major planned updates:
	- fit orbits of multiple objects in one system
	- fit Gaia astrometry & RVs
	- marginalize over instrumental uncertainties

Changelog:
++++++++++

**1.0.0 (2018-10-29?)**

- Initial release.