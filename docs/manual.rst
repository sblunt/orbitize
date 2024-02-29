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
    with use and explanation of the coordinate system by Sarah Blunt.

In its “standard” mode, ``orbitize!`` assumes that the user only has relative astrometric data to fit. 
In the ``orbitize!`` coordinate system, relative R.A. and declination can be expressed as the following functions 
of orbital parameters 

.. math::
    \Delta R.A. = \pi a(1-ecosE)[cos^2{i\over 2}sin(f+\omega_p+\Omega)-sin^2{i\over 2}sin(f+\omega_p-\Omega)]

    \Delta decl. = \pi a(1-ecosE)[cos^2{i\over 2}cos(f+\omega_p+\Omega)+sin^2{i\over 2}cos(f+\omega_p-\Omega)]

where 𝑎, 𝑒, :math:`\omega_p`, Ω, and 𝑖 are orbital parameters, and 𝜋 is the system parallax. f is
the true anomaly, and E is the eccentric anomaly, which are related to elapsed time
through Kepler’s equation and Kepler’s third law

.. math::
    M = 2\pi ({t\over P}-(\tau -\tau_{ref}))

    ({P\over yr})^2 =({a\over au})^3({M_\odot \over M_{tot}})
    
    M =E-esinE 
    
    f = 2tan^{-1}[\sqrt{{1+e\over 1-e}}tan{E\over 2}]

``orbitize!`` employs two Kepler solvers to convert between mean
and eccentric anomaly: one that is efficient for the highest eccentricities, and Newton’s method, which in other cases is more efficient for solving for the average
orbit. See `Blunt et al. (2020) <https://iopscience.iop.org/article/10.3847/1538-3881/ab6663>`_ for more detail.


From scrutinizing the above sets of equations, one may observe
a few important degeneracies:

1. Individual component masses do not show up anywhere in this equation set.

2. The degeneracy between semimajor axis 𝑎, total mass :math:`𝑀_{tot}`, and
parallax 𝜋. If we just had relative astrometric measurements and no external knowledge of the system parallax, 
we would not be able to distinguish between a system
that has larger distance and larger semimajor axis (and therefore larger total mass,
assuming a fixed period) from a system that has smaller distance, smaller semimajor
axis, and smaller total mass. 

3. The argument of periastron :math:`\omega_p` and the position angle of nodes Ω. 
The above defined R.A. and decl. functions are invariant to the transformation:

.. math::
    \omega_p' = \omega_p + \pi
    
    \Omega' = \Omega - \pi

which creates a 180◦ degeneracy between particular values of :math:`\omega_p` and Ω, and
a characteristic “double-peaked” structure in marginalized 1D posteriors of these
parameters. 

Solutions to breaking degeneracies 1 and 3 can be found in the next section. 


Using Radial Velocities 
+++++++++++++++++
In the ``orbitize!``coordinate system, and relative to the system barycenter, the
radial velocity of the planet due to the gravitational influence of the star is:
.. math::
    rv_p(f) = [\sqrt{{G\over (1-e**2)}}]M_* sini(M_{tot})^{-1/2}a^{-1/2}(cos(\omega_p+f)+ecos\omega_p)

    rv_*(f) = [\sqrt{{G\over (1-e**2)}}]M_p sini(M_{tot})^{-1/2}a^{-1/2}(cos(\omega_*+f)+ecos\omega_*)

where 𝜔∗ is the argument of periastron of the star’s orbit, which is equal to 𝜔𝑝 +
180◦.

In these equations, the individual component masses m𝑝 and m∗ enter. This means
radial velocity measurements break the total mass degeneracy and enable measurements of individual component masses 
(“dynamical” masses). However it is crucial to keep in mind that radial velocities of a planet do not enable 
dynamical mass measurements of the planet itself, but of the star. 
Radial velocity measurements also break the Ω/𝜔 degeneracy discussed in the
previous section, uniquely orienting the orbit in 3D space.

``orbitize!``can perform joint fits of RV and astrometric data in two different
ways, which have complementary applications. 

The first method is automatically triggered when an
``orbitize!``user inputs radial velocity data. ``orbitize!``automatically parses
the data, sets up an appropriate model, then runs the user’s Bayesian computation
algorithm of choice to jointly constrain all free parameters in the fit. ``orbitize!``
can handle both primary and secondary RVs, and fits for the appropriate dynamical
masses when RVs are present; when primary RVs are included, ``orbitize!``fits for
the dynamical masses of secondary objects, and vice versa. 
Instrumental nuisance parameters (RV zeropoint offset, 𝛾, and white noise jitter, 𝜎) for each RV instrument
are also included as additional free parameters in the fit if the user specifies different
instrument names in the data file.

The second method of jointly fitting RV and astrometric data in ``orbitize!`` separates out the fitting of 
radial velocities and astrometry, enabling a user to fit “one at a
time,” and combine the results in a Bayesian framework. First, a user performs a
fit to just the radial velocity data using, for example, radvel (but can be any radial
velocity orbit-fitting code). The user then feeds the numerical posterior samples
into ``orbitize!`` through the ``orbitize.priors.KDEPrior`` object. This prior
creates a representation of the prior using kernel density estimation 
(`kernel density estimation <https://mathisonian.github.io/kde/>`_),
which can then be used to generate random prior samples or compute the prior
probability of a sample orbit. Importantly, this prior preserves covariances between
input parameters, allowing ``orbitize!``to use an accurate representation of the RV
posterior to constrain the fit. This method can be referred to as the “posteriors as priors”
method, since posteriors output from a RV fitting code are, through KDE sampling,
being applied as priors in``orbitize!``.


More coming soon!