.. _manual:

orbitize! Manual
==============

Intro to ``orbitize!``
+++++++++++++++++

``orbitize!`` hinges on the two-body problem, which describes the paths of two
bodies gravitationally bound to each other as a function of time, 
given parameters determining the position and velocity of both objects at a particular epoch.
There are many basis sets (orbital bases) that can be used to describe an orbit, 
which can then be solved using Kepler’s equation, but first it is important to be explicit about coordinate systems. 

.. Note:: 
    For an interactive visualization to define and help users understand our coordinate system, 
    you can check out `this GitHub tutorial <https://github.com/sblunt/orbitize/blob/main/docs/tutorials/show-me-the-orbit.ipynb>`_.
    
    There is also a `YouTube video <https://www.youtube.com/watch?v=0e24VUhQmbM>`_  
    with use and explanation of the coordinate system.

In its “standard” mode, ``orbitize!`` assumes that the user only has relative astrometric data to fit. 
In the ``orbitize!`` coordinate system, relative R.A. and declination can be expressed as the following functions 
of orbital parameters 

.. math::
    \Delta R.A. = \pi a(1-ecosE)[cos^2{i\over 2}sin(f+\omega_p+\Omega)-sin^2{i\over 2}sin(f+\omega_p-\Omega)]

    \Delta decl. = \pi a(1-ecosE)[cos^2{i\over 2}cos(f+\omega_p+\Omega)-sin^2{i\over 2}cos(f+\omega_p-\Omega)]

where 𝑎, 𝑒, :math:`\omega_p`, Ω, and 𝑖 are orbital parameters, and 𝜋 is the system parallax. f is
the true anomaly, and E is the eccentric anomaly, which are related to elapsed time
through Kepler’s equation and Kepler’s third law

.. math::
    M = 2\pi ({t\over P}-(\tau -\tau_ref))

    ({P\over yr})^2 =({a\over au})^3({M_\odot \over M_tot})

    M =E-esinE

    f = 2tan^-1[\sqrt{{1+e\over 1-e}}tan{E\over 2}]

``orbitize!`` employs two Kepler solvers to convert between mean
and eccentric anomaly: one that is efficient for the highest eccentricities, and Newton’s method, which in other cases is more efficient for solving for the average
orbit. See `Blunt et al. (2020) <https://iopscience.iop.org/article/10.3847/1538-3881/ab6663>`_ for more detail.


From scrutinizing the above sets of equations, one may observe
a few important inherent degeneracies. 

First, notice that the individual component masses do not show up anywhere in this equation set. 
While it is impossible to measure dynamical masses for either the primary or the secondary using just
relative astrometry, there are methods to constrain the system. 
If the mass of the planet can be safely assumed to be negligible compared to the mass of the star, 
then the total mass derived from Keplerian analysis can be treated as a constraint on the dynamical mass 
of the primary. 
In practice, the reverse logic is often employed: an independent constraint
on the mass of the primary (from e.g., spectroscopic analysis) is used as a prior on
the total mass when the planet mass is small and can be ignored.

A second important degeneracy is between semimajor axis 𝑎, total mass :math:`𝑀_{tot}`, and
parallax 𝜋. If we just had relative astrometric measurements and no external knowledge of the system parallax, 
we would not be able to distinguish between a system
that has larger distance and larger semimajor axis (and therefore larger total mass,
assuming a fixed period) from a system that has smaller distance, smaller semimajor
axis, and smaller total mass. Luckily, we live in an era where parallax measurements
are excellent overall thanks to the Gaia mission, with which strict priors can often be applied to, 
breaking the degeneracy and enabling dynamical mass measurements of stars (when planet mass is negligible). 
However, this degeneracy is important to take into account when considering the impact of potential
biases in parallax or stellar mass measurements. 

A final degeneracy I would like to point out concerns the argument of periastron :math:`\omega_p`
and the position angle of nodes Ω. The above defined R.A. and decl. functions are invariant to the transformation:

.. math::
    \omega_p' = \omega_p + \pi

    \Omega' = \Omega - \pi

which creates a 180◦ degeneracy between particular values of :math:`\omega_p` and Ω, and
a characteristic “double-peaked” structure in marginalized 1D posteriors of these
parameters (see Figure 4.2 for an example). 
Physically, this degeneracy comes about
because relative astrometry alone only constrains motion in the plane of the sky; an
orbit tilted toward the observer, with the planet moving away from the observer has
the same projection on the plane of the sky as an orbit tilted away from the observer,
with the planet moving toward the observer. In practice, this degeneracy is handy,
because if the :math:`\omega_p`/Ω posteriors do not appear identical before and after 180◦, 
it is generally an indication that the MCMC chains are unconverged.