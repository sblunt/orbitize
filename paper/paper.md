---
title: 'orbitize! v3: Orbit-fitting for the High-contrast Imaging Community'
tags:
  - Python
  - astronomy
  - orbit-fitting
  - exoplanets
  - high-contrast imaging
authors:
  - name: Sarah Blunt
    orcid: 0000-0002-3199-2888
    corresponding: true
    affiliation: "1,2"
  - name: Jason Wang
    orcid: 0000-0003-0774-6502
    affiliation: "1"
  - name: Lea Hirsch
    affiliation: 
  - name: Roberto Tejada
    affiliation: 
  - name: Vighnesh Nagpal
    affiliation: 
  - name: Tirth Dharmesh Surti
    affiliation: 
  - name: Sofia Covarrubias
    affiliation: 
  - name: Thea McKenna
    affiliation: 
  - name: Rodrigo Ferrer Chávez
    affiliation: 
  - name: Jorge Llop Sayson
    affiliation: 
  - name: Mireya Arora
    affiliation: 
  - name: Amanda Chavez
    affiliation: 
  - name: Devin Cody
    affiliation: 
  - name: Saanika Choudhary
    affiliation: 
  - name: Adam Smith
    affiliation: 
  - name: William Balmer
    affiliation: 
  - name: Thomas Stolker
    affiliation: 
  - name: Hannah Gallamore
    affiliation: 
  - name: Clarissa Do Ó
    affiliation: 
  - name: Eric Nielsen
    affiliation: 
  - name: Robert de Rosa
    affiliation: 

affiliations:
 - name: Center for Interdisciplinary Exploration and Research in Astrophysics (CIERA), Northwestern University
   index: 1
 - name: California Institute of Technology
   index: 2

date: 1 March 2024
bibliography: paper.bib

---

# Summary

`orbitize!` is a package for Bayesian modeling of the orbital parameters of resolved binary 
objects from timeseries measurements. It was developed with the needs of the high-contrast
imaging community in mind, and has since also become widely used in the binary star community.
A generic `orbitize!` use case involves translating relative astrometric timeseries, optionally 
combined with radial velocity or astrometric timeseries, into a set of derived orbital posteriors.

This paper is published alongside the release of `orbitize!` version 3.0, which 
has seen significant expansions in functionality and accessibility since the 
release of version 1.0 [@Blunt:2020].

# Statement of need

The orbital parameters of directly-imaged planets and binary stars can tell us about
their present-day dynamics and formation histories [@Bowler:2016], as well as about 
their inherent physical characteristics (particularly mass, generally called ``dynamical 
mass'' when derived from orbital constraints, e.g. [@Brandt:2021], [@Lacour:2021]). 

`orbitize!` is used widely in the imaged exoplanet and binary star communities for 
translating astrometric data to information about eccentricities [@Bowler:2020], obliquities [@Bryan:2020], 
dynamical masses [@Lacour:2021], and more. 

Each new released version of the `orbitize!` source code is automatically archived on Zenodo [@orbitize].

# Major features added since v1

For a detailed overview of the `orbitize!` API, core functionality (including information 
about our Kepler solver), and initial verification, we refer readers to [@Blunt:2020]. 
This section lists major new features that have been added to the 
code since the release of version 1.0 and directs readers to more information about each.
A complete descriptive list of modifications to the code is maintained in our 
[changelog](https://orbitize.readthedocs.io/en/latest/#changelog).

Major new features since v1 include:

1. The ability to jointly fit radial velocity (RV) timeseries, both RVs of the secondary 
    companion (see Section 3 of [Blunt:2023a]) and RVs of the primary
    star. RVs of the primary star can either be passed into `orbitize!` directly (see the [radial velocity tutorial](https://orbitize.readthedocs.io/en/latest/tutorials/RV_MCMC_Tutorial.html)), or fit separately and passed in as prior
    information (see the [non-orbitize! posteriors as priors tutorial](https://orbitize.readthedocs.io/en/latest/tutorials/Using_nonOrbitize_Posteriors_as_Priors.html).)

2. The ability to jointly fit absolute astrometry of the primary star. `orbitize!` can fit
    the Hipparcos-Gaia catalog of accelerations [@Brandt:2021] (see the [HGCA tutorial](https://github.com/sblunt/orbitize/blob/v3/docs/tutorials/HGCA_tutorial.ipynb)), as well as Hipparcos intermediate astrometric data and Gaia 
    astrometry, following [@Nielsen:2020] (see the [Hipparcos IAD tutorial](https://orbitize.readthedocs.io/en/latest/tutorials/Hipparcos_IAD.html)). It can also handle arbitrary absolute astrometry (Sarah to add tutorial link).

3. In addition to the MCMC and OFTI posterior computation algorithms documented in [@Blunt:2020], 
    `orbitize!` version 3 also implements a nested sampling backend, via `dynesty` [@Speagle:2020] 
    (see the [`dynesty` tutorial](https://github.com/sblunt/orbitize/blob/dynesty/docs/tutorials/dynesty_tutorial.ipynb).)

4. `orbitize!` version 3 implements two prescriptions for handling multi-planet effects. 
    Keplerian epicyclic motion of the primary star due to multiple orbiting bodies, 
    following [@Lacour:2021], is discussed in the [multi-planet tutorial](https://orbitize.readthedocs.io/en/latest/tutorials/Multiplanet_Tutorial.html), and N-body interactions are discussed in [@Covarrubias:2022]. The Keplerian epicyclic motion
    prescription only accounts for star-planet interactions, treating the motion of the star as a sum of Keplerians, 
    while the N-body prescription models this effect as well as planet-planet interactions.

5. The ability to fit in different orbital bases [@Surti:2023], [@Ferrer-Chavez:2021] (see the 
    [changing basis](https://orbitize.readthedocs.io/en/latest/tutorials/Changing_bases_tutorial.html) tutorial).

# Verification and Documentation

`orbitize!` implements a full stack of automated testing and documentation building 
practices. We use GitHub Actions to automatically run a suite of unit tests, maintained in orbitize/tests,
each time code is committed to the public repository or a pull request is opened. The jupyter notebook
tutorials, maintained in orbitize/docs/tutorials, are also automatically run when a 
pull request to the `main` branch is opened. Documentation is built using `sphinx`, and hosted
on readthedocs.org at [orbitize.info](https://orbitize.readthedocs.io/en/latest/). We also
maintain a set of longer-running tests in orbitize/tests/end-to-end-tests that show real
scientific use cases of the code. These tests are not automatically run.

`orbitize!` is documented through API docstrings describing individual functions, which are accessible on [our readthedocs page](https://orbitize.readthedocs.io/en/latest/api.html), a set of [jupyter notebook tutorials](https://orbitize.readthedocs.io/en/latest/tutorials.html) walking the user through a particular application, a set of [frequently asked questions](https://orbitize.readthedocs.io/en/latest/faq.html),
and an in-progress [``manual,''](https://orbitize.readthedocs.io/en/orbitize-manual/manual.html) describing orbit-fitting with `orbitize!` from first principles.

# Comparison to Similar Packages

Since the release of `orbitize!` version 1, other open-source packages have been released that have 
similar goals to `orbitize!`, notably `orvara` and `octofitter`. This is a wonderful development, as 
each package has benefitted from open sharing of knowledge. `orbitize!`, `orvara`, and `octofitter` can 
do many similar things, but each has unique features and strengths; as an example, `octofitter` is 
extraordinarily fast, and enables joint astrometry extraction and orbit modeling, while `orbitize!` has unique 
abilities to fit arbitrary absolute astrometry (i.e. not from Hipparcos or Gaia) and model data using an N-body backend. 
`orvara` analytically marginalizes over parallax assuming a prior informed by Gaia, a significant speed advantage, while 
`orbitize!` allows different parallax priors to be used. We recommend users of each package compare the implementations 
of the particular features they wish to use. 

Best practices for orbit-fitting, particularly using radial velocities, for which treatment of stellar 
activity is an active area of research, and absolute astrometry with Gaia and Hipparcos, for which
correlated error treatment is an active area of research, evolve quickly. The philosophy of `orbitize!`
is to, as much as possible, implement multiple approaches to a problem, evidenced by our multiple
implementations of radial velocity joint fitting and absolute astrometry joint fitting (described above). 
For detailed information about our particular implementations, we direct the reader to our documentation. 

# Acknowledgements

Our team gratefully acknowledges support the Heising-Simons Foundation.  S.B. and J.J.W. are supported 
by NASA Grant 80NSSC23K0280. 

# References