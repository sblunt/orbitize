---
title: 'orbitize! v4: Joint Orbit Fitting with Gaia DR4 Astrometry and Flexible Analytic Linear Marginalization'
tags:
- Python
- astronomy
- exoplanets
- astrometry
- orbit fitting
- Gaia
authors:
- name: Clarissa R. Do Ó
  given-names: Clarissa R.
  surname: Do Ó
  orcid: 0000-0001-5173-2947
  corresponding: true
  affiliation: '1'
- name: Eshel Dror
  affiliation: '2'
- name: Sarah Blunt
  orcid: 0000-0002-3199-2888
  affiliation: '3'
- name: Vanessa P. Bailey
  orcid: 0000-0002-5407-2806
  affiliation: '4'
- name: William O. Balmer
  orcid: 0000-0001-6396-8439
  affiliation: '5'
- name: Beth A. Biller
  orcid: 0000-0003-4614-7035
  affiliation: '6'
- name: Quinton Blackston
  affiliation: '2'
- name: Aniruddh Chalagulla
  affiliation: '2'
- name: Robert J. De Rosa
  given-names: Robert J.
  surname: De Rosa
  orcid: 0000-0002-4918-0247
  affiliation: '7'
- name: Chih-Chun Hsu
  orcid: 0000-0002-5370-7494
  affiliation: '5'
- name: Isabela Huckabee
  affiliation: '8'
- name: Farrah Molina
  affiliation: "5,9"
- name: Niklas Naworal
  affiliation: '2'
- name: Eric L. Nielsen
  affiliation: '10'
- name: Tomas Stolker
  orcid: 0000-0002-5823-3072
  affiliation: '11'
- name: Rifah Tasnim
  affiliation: '8'
- name: David Trevascus
  affiliation: '12'
- name: Edward Turner
  affiliation: '6'
- name: Vidya Venkatesan
  affiliation: '13'
- name: Jason J. Wang
  orcid: 0000-0003-0774-6502
  affiliation: "5,15"
- name: Marie Ygouf
  affiliation: '4'
- name: Jingwen Zhang
  orcid: 0000-0002-2696-2406
  affiliation: '14'
affiliations:
- name: Department of Astronomy, California Institute of Technology, Pasadena, CA, USA
  index: 1
- name: Department of Astronomy and Astrophysics, University of California, Santa Cruz, Santa Cruz, CA, USA
  index: 2
- name: Department of Astrophysical and Planetary Sciences, University of Colorado Boulder, Boulder, CO, USA
  index: 3
- name: Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA, USA
  index: 4
- name: Center for Interdisciplinary Exploration and Research in Astrophysics (CIERA), Northwestern University, Evanston, IL, USA
  index: 5
- name: Institute for Astronomy, School of Physics and Astronomy, University of Edinburgh, Edinburgh,
    UK
  index: 6
- name: European Southern Observatory, Santiago, Chile
  index: 7
- name: Department of Astronomy and Carl Sagan Institute, Cornell University, Ithaca, NY, USA
  index: 8
- name: Department of Physics and Engineering, University of St. Thomas, Houston, TX, USA
  index: 9
- name: Department of Astronomy, New Mexico State University, Las Cruces, NM, USA
  index: 10
- name: Leiden Observatory, Leiden University, Leiden, The Netherlands
  index: 11
- name: Max Planck Institute for Astronomy, Heidelberg, Germany
  index: 12
- name: University of California, Riverside, Riverside, CA, USA
  index: 13
- name: Department of Physics, University of California, Santa Barbara, Santa Barbara, CA, USA
  index: 14
- name: Department of Physics and Astronomy Northwestern University, Evanston, IL 60208, USA
  index: 15
date: 19 November 2026
bibliography: paper.bib
---

# Summary

`orbitize!` is a Python package for Bayesian orbit fitting of planetary and stellar
companions [@Blunt:2020]. It provides a unified framework for fitting relative
astrometry, radial velocities, and absolute astrometry, supporting analyses of both
resolved and unresolved systems. Available posterior sampling methods include
parallel-tempered Markov chain Monte Carlo (MCMC) through `ptemcee` [@emcee; @ptemcee]
and nested sampling through `dynesty` [@Speagle:2020], `nautilus` [@nautilus], and
`pymultinest` [@pymultinest; @multinest]. This paper accompanies the release of
`orbitize!` version 4.0 and describes the functionality added since version 3.0
[@{2024JOSS....9.6756B}], with a particular focus on joint orbit fitting with Gaia DR4
epoch astrometry and the implementation of the Orbits for the Extremely Impatient
(OFTeI) algorithm (Blunt et al. 2026b, in preparation).

# Statement of need

The fourth data release (DR4) of the *Gaia* mission [@{2016A&A...595A...1G}] will offer
new opportunities to measure companion masses and characterize planetary system
architectures through epoch astrometry:. the individual measurements of a source's
position along Gaia's scan direction at different observation times. These measurements
will offer new opportunities to characterize planetary system architectures. Many
companions detected through Gaia astrometry are also expected to be detectable with
other methods, particularly radial velocities [@LammersWinn:2026; @Winn:2022]. Combining
astrometry with stellar radial velocities can break the mass--inclination degeneracy of
RV-only fits and constrain three-dimensional orbital orientations. For directly imaged
systems, relative astrometry can only constrain the total system mass. Adding the
primary's absolute orbital motion can constrain the individual component masses.

`orbitize!` v4 provides these capabilities in an open source framework, by modeling
individual Gaia along-scan measurements and fitting them alone or jointly with stellar
radial velocities, companion radial velocities, and relative astrometry. Additionally,
implementation of the Orbits for the Extremely Impatient Algorithm, a novel analytic
marginalization scheme for relative and absolute astrometry (Blunt et al. 2026b, in
prep), together with an optimized Kepler solver, enables much faster convergence
relative to earlier versions of the code.

## Gaia DR4 epoch astrometry

Version 4 introduces the `gaia.DR4LogProb` class for fitting Gaia epoch astrometry. The
forward model includes five astrometric parameters: two position offsets, $\alpha_{*,0}$
and $\delta_0$, two proper motion components, $\mu_{\alpha*}$ and $\mu_\delta$, and
parallax, $\varpi$. Together with the primary's orbital reflex motion about the system
barycenter, these describe its apparent motion on the sky [@Nielsen:2020; @Brandt:2024].
Users can choose among the orbital fitting bases supported by `orbitize!` [see
@Blunt:2020; @Blunt:2023a, for orbital parameter definitions]. The predicted sky motion
is then projected onto the scan direction at each Gaia observation. For measurement $i$,
the predicted along-scan position is

$$
\begin{aligned}
\eta_i ={}&
\left(\alpha_{*,0} + \mu_{\alpha*}\Delta t_i
      + x_{\star,i}\right)\sin\psi_i \\
&+
\left(\delta_0 + \mu_{\delta}\Delta t_i
      + y_{\star,i}\right)\cos\psi_i
+ \varpi \Pi_i ,
\end{aligned}
$$

where $\psi_i$ is the scan position angle, $\Pi_i$ is the along-scan parallax factor,
and $\varpi$ is the parallax. The proper motion components are
$\mu_{\alpha*} = \dot{\alpha}\cos\delta$ in right ascension and
$\mu_\delta = \dot{\delta}$ in declination, both expressed in mas yr$^{-1}$. The
position offsets $\alpha_{*,0}$ and $\delta_0$ are defined at a reference epoch (by
default the Gaia DR4 reference epoch of
[J2017.5](https://esa.github.io/gaia-supdate/gaiasupdate.html#gaiasupdate.constants.DR4_REFERENCE_EPOCH)),
and $\Delta t_i$ is the elapsed time in Julian years. The quantities $x_{\star,i}$ and
$y_{\star,i}$ are the photocenter's barycentric orbital offsets in right ascension,
including the $\cos\delta$ factor, and declination, respectively. For systems with
multiple companions, these offsets include their combined reflex signal, and orbital
parameters are defined in Jacobi coordinates [@Lacour:2021].

The astrometric log-likelihood is

$$
\ln \mathcal{L}_{\mathrm{DR4}}
=
-\frac{1}{2}\sum_i
\left[
\frac{(d_i-\eta_i)^2}{\sigma_i^2+s_{\mathrm{G}}^2}
+
\ln\!\left(2\pi[\sigma_i^2+s_{\mathrm{G}}^2]\right)
\right],
$$

where $d_i$ and $\sigma_i$ are the measured along-scan position and its reported
uncertainty. $\eta_i$ denotes the predicted along-scan angular offset for observation
$i$ in milliarcseconds. An additional jitter term, $s_{\mathrm{G}}$, can be fixed or
fitted and is added in quadrature to the reported uncertainties, as in
@{2023A&A...674A..10H}. Position offsets and proper motions are sampled explicitly
alongside the orbital parameters and parallax with configurable priors.

Gaia measurements can be fitted alone or combined with the primary radial velocities,
companion radial velocities, relative astrometry, and the Hipparcos IAD [@Nielsen:2020].
All observations are predicted from a shared orbital model, and since we assume the
datasets are independent, we add the log-likelihoods for all data. Priors on shared
parameters enter the posterior once. The resulting posterior is sampled with the user's
choice of posterior sampling algorithm (e.g. MCMC, nested sampling), allowing joint
constraints on orbital and astrometric parameters. `orbitize!` automatically sets up the
parameters required by the supplied datasets, and users can select the orbital fitting
basis and customize the priors before sampling.

# Major features added since v3

For a detailed overview of the `orbitize!` API and core functionality, we refer the
reader to @Blunt:2020. This section highlights the major features added since version
3.0. For a complete list of changes, see the
[changelog](https://orbitize.readthedocs.io/en/latest/#changelog). Major new features
since v3 include:

1.  A likelihood for Gaia DR4 epoch astrometry, implemented in `gaia.DR4LogProb`. The
    model projects the primary's motion onto each observation's scan direction. An
    optional astrometric jitter term, either fixed or fitted, is added in quadrature to
    the reported measurement uncertainties. Gaia measurements can be fitted alone or
    jointly with other datasets.

2.  The `simulate_dr4` module generates mock Gaia DR4 along-scan measurements from a
    specified orbital solution and [Gaia Observation Forecast Tool
    (GOST)](https://gaia.esac.esa.int/gost/) scanning predictions. The simulations
    include stellar reflex motion, proper motion, parallax, and Gaussian noise drawn
    using a magnitude-dependent uncertainty model adopted from
    [Gaiamock](https://github.com/kareemelbadry/gaiamock) [@ElBadry:2024], based on the
    precision curves presented by @{2023A&A...674A..10H}. The resulting tables can be
    passed directly to the DR4 likelihood for injection recovery tests.

3.  Worked examples demonstrate the new fitting and plotting capabilities. The [Gaia BH3
    Tutorial](https://orbitize.readthedocs.io/en/latest/tutorials/Gaia_DR4_BH3.html)
    demonstrates fitting Gaia epoch astrometry alone for the black hole binary
    discovered by @{2024A&A...686L...2G}. The [Gaia-4
    Tutorial](https://orbitize.readthedocs.io/en/latest/tutorials/Gaia_DR4_Gaia4.html)
    demonstrates a joint fit with stellar radial velocities from @{2025AJ....169..107S},
    while the [HD 114762
    Tutorial](https://orbitize.readthedocs.io/en/latest/tutorials/Gaia_DR4_EAS.html)
    provides another example of joint RV and Gaia epoch astrometry fitting. These three
    tutorials use epoch astrometry from the [June 2026 Gaia DR4
    prerelease](https://www.cosmos.esa.int/web/gaia/dr4-prerelease). The [$\mu$ Ara
    Tutorial](https://orbitize.readthedocs.io/en/latest/tutorials/HD160691_Joint_Fit.html)
    demonstrates generating mock Gaia DR4 astrometry for a four-planet system and
    jointly fitting it with observed stellar radial velocities. All four examples
    illustrate posterior sampling and visualization of the inferred orbits.

4.  The `marginalizer` module provides a framework for implementation of analytic
    marginalization schemes in `orbitize!`. The Orbits for the Extremely Impatient
    (OFTeI) algorithm, implemented in `marginalizer.OFTeI_seppa`, allows partial linear
    marginalization over the semimajor axis and position angle of nodes for systems with
    one secondary and relative astrometry data. Similarly, the algorithm implemented in
    `marginalizer.OFTeI_DR4_gaia` performs partial linear marginalization over proper
    motion, position offset, parallax, semimajor axis, and position angle of nodes for
    systems with one secondary and astrometric data from Gaia DR4.

5.  The optimization of the `kepler` module significantly speeds up the solving of
    kepler's equations by migrating more of the code to C. The improvement is most
    significant in evaluating a single set of parameters. For samplers such as Markov
    Chain Monte Carlo which evaluate one set of parameters at a time, the optimized
    sampler is 12.39 times faster, with the calculation done by the `kepler` module
    taking about 2% of the previous time. For evaluating an array of 30,000 sets of
    parameters, the optimized calculation takes about 65% of the previous time.

6.  Refactoring of the functions in the `plot` module into the `plot.Plotter` class
    facilitates clearer code flow and new plotting capabilities. The `plot.Plotter`
    class precomputes shared attributes of the different plotting methods when
    initialized, so that they are not recomputed on each call of a plotting method. The
    `plot.Plotter.plot_orbits` method has new capabilities in plotting the orbits of
    multiple bodies, displaying correlated errorbars on Right Ascension/Declination
    data, and customizing grouping of sources of error for Radial Velocity.
    `plot.Plotter.plot_residuals` similarly supports multiple bodies and the separation
    of errors. Multiple tutorials, especially the [Advanced Plotting
    Tutorial](https://github.com/sblunt/orbitize/blob/v4/docs/tutorials/Plotting_tutorial.ipynb),
    demonstrate new plotting capabilities and many new customization options.

7.  Joint fitting for reflected-light photometry with Lambertian disk approximation,
    following the implementation outlined in [@{2026RNAAS..10..203M}]. This
    implementation adopts relative brightness
    $\mathrm{(brightness)} \propto \frac{A\alpha}{R^2}$ where $A$ is Albedo; $\alpha$ is
    the angle between the star, the planet, and the observer; and $R$ is the distance
    between the planet and star. [^1]

8.  Support for two importance nested sampling algorithms. The sampler implemented in
    `sampler.NautilusSampler` uses the `Nautilus-Sampler` package and determines the
    boundaries of nested sampling shells using deep learning [@nautilus]. The
    `sampler.MultiNest` sampler uses the `pymultinest` package
    [@pymultinest; @multinest].

# Verification and Documentation

For details of the automated testing and documentation infrastructure, see the
"Verification and Documentation" section of @{2024JOSS....9.6756B}.

# Comparison to Similar Packages

For an overview of related orbit fitting packages, we refer readers to
@{2024JOSS....9.6756B}. Current capabilities and examples for `orbitize!` are described
in our [documentation](https://orbitize.readthedocs.io/en/latest/). Different orbit
fitting packages offer different approaches to model construction and posterior
inference. `orbitize!` v4 introduces a generic framework for marginalizing over linear
model parameters, reducing the dimensionality of the sampled parameter space. `kima`
[@Baycroft:2026] provides the `GAIAmodel` and `RVGAIAmodel` for fitting Gaia epoch
astrometry alone or jointly with radial velocities, using diffusive nested sampling.
`orvara` [@orvara] analytically marginalizes over parallax, barycentric proper motion,
and instrument-specific radial velocity zero points. `octofitter` [@Thompson:2023]
supports gradient-based inference using the No-U-Turn variant of Hamiltonian Monte Carlo
and can also fit Gaia DR4 epoch astrometry, as illustrated in its [prerelease-data
tutorial](https://sefffal.github.io/Octofitter.jl/stable/gaia-dr4-prerelease/), which
includes Gaia BH3. The `exoplanet` toolkit [@exoplanet] provides tools for building and
fitting models of transit and radial velocity data using the Bayesian modeling package
`PyMC`. A detailed performance comparison will be presented in a forthcoming paper (Dror
et al., in preparation). `ExoPlanetLab` [@Hobbs:2026] also supports joint modeling of
Gaia epoch astrometry, radial velocities, and relative astrometry, with multiple
posterior sampling methods and a Student-$t$ likelihood.

# Acknowledgements

C.D.O acknowledges support from the B. Thomas Soifer Fellowship at the California
Institute of Technology.

This paper describes additions made to `orbitize!` between versions 3.0.0 and 4.0.0. We
are extremely grateful to Isabel Angelo, Henry Ngo, James Graham, Logan Pearce, and
Malena Rice, who contribtued code to version 1.0.0, and are included as authors in
@Blunt:2020. We are also grateful for the useful discussions with William Thompson.

# References

[^1]: In [@{2026RNAAS..10..203M}], equation A5 states that R is the planetary radius,
    which in this context means the distance between the planet and the star (see their
    equation A1)
