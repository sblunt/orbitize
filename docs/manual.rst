.. _manual:

orbitize! Manual
==============

Intro to ``orbitize!``
+++++++++++++++++
Start with Section 4.2 of Sarah's thesis: https://thesis.library.caltech.edu/16076/

At its core, ``orbitize!`` turns data into orbits. 
This is done when relative kinematic measurements of a primary and secondary body are converted to posteriors over 
orbital parameters through Bayesian analysis.

``orbitize!`` hinges on the two-body problem, which describes the paths of two
bodies gravitationally bound to each other. 
The solution of the two-body problem describes the motion of each body as a 
function of time, given parameters determining the position and velocity of both objects at a particular epoch. 



There are many basis sets (orbital bases) that can be used to describe an orbit, 
which can then be solved using Kepler’s equation. 

It is important, then, to be explicit about coordinate systems. 

For an interactive visualization to define and help users understand our coordinate system, 
you can check out `this GitHub tutorial <https://github.com/sblunt/orbitize/blob/main/docs/tutorials/show-me-the-orbit.ipynb>`_.

There is also a `YouTube video <https://www.youtube.com/watch?v=0e24VUhQmbM>`_. 
with use and explaination of the coordinate system.

In its “standard” mode, ``orbitize!`` assumes that the user only has relative astrometric data to fit. 
To obtain these measurements, an astronomer takes an image containing two point sources 
and measures the position of the planet relative to the star in angular coordinates. 
In the ``orbitize!`` coordinate system, relative R.A. and decl. can be expressed as the following functions 
of orbital parameters 

.. math::
    \delta R.A. = \pi a(1-ecosE)[cos^2{i\over 2}sin(f+\omega_p+\Omega)-sin^2{i\over 2}sin(f+\omega_p-\Omega)] $$
    \delta decl. = \pi a(1-ecosE)[cos^2{i\over 2}cos(f+\omega_p+\Omega)-sin^2{i\over 2}cos(f+\omega_p-\Omega)] $$

where 𝑎, 𝑒, 𝜔p, Ω, and 𝑖 are orbital parameters, and 𝜋 is the system parallax. f is
the true anomaly, and E is the eccentric anomaly, which are related to elapsed time
through Kepler’s equation and Kepler’s third law: