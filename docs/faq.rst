.. _faq:

Frequently Asked Questions
==========================

Here are some questions we get often. Please suggest more by raising an 
issue in the `Github Issue Tracker <https://github.com/sblunt/orbitize/issues>`_.

**What does this orbital parameter mean?**

We think the best way to understand the orbital parameters is to see how they affect the orbit visually.
Play around with this `interactive orbital elements notebook <https://github.com/sblunt/orbitize/blob/main/docs/tutorials/show-me-the-orbit.ipynb>`_ 
(you'll need to run on your machine).

**What is τ and how is it related to epoch of periastron?**

We use τ to define the epoch of periastron as we do not know when periastron will be for many of our directly
imaged planets. A detailed description of how τ is related to other quantities such as :math:`t_p` is available:

.. toctree::
   :maxdepth: 1

   faq/Time_Of_Periastron.ipynb


**Why is the default prior on inclination a sine prior?**

Our goal with the default prior is to have an isotropic distribution of the orbital plane.
To obtain this, we use inclination and position angle of the ascending node (PAN) to
define the orbital plane. They actually coorespond to the two angles in a spherical coordinate system:
inclinaion is the polar angle and PAN is the azimuthal angle. Becuase of this choice of coordinates,
there are less orbital configurations near the poles (when inclination is near 0 or 180), so we use
a sine prior to downweigh those areas to give us an isotropic distribution. 
A more detailed discussion of this is here:

.. toctree::
   :maxdepth: 1

   faq/Orientation_Of_Orbit.ipynb