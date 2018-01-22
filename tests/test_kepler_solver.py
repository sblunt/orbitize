"""
Test the orbitize.kepler module which solves for the orbits of the planets
"""
import pytest
import numpy as np
import orbitize.kepler as kepler

# need to calcualte to a 100 microarcsecs. Should improve this in the future.
threshold = 1e-4

def test_orbit_e03():
    """
    Test orbitize.kepler.calc_orbit() by comparing this code to the output of James Graham's code which has been used in
    many published papers. Note that orbitize currently uses Rob De Rosa's eccentricity solver. The two are not guaranteed to be consistent
    below 100 microarcseconds

    Pretty standard orbit with ecc = 0.3
    """
    # sma, ecc, tau, argp, lan, inc, plx, mtot
    orbital_params = (10, 0.3, 0.3, 0.5, 1.5, 3, 50, 1.5)
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, orbital_params[0], orbital_params[1], orbital_params[2], orbital_params[3],
                                        orbital_params[4], orbital_params[5], orbital_params[6], orbital_params[7])

    true_raoff = [0.15286786,  0.18039408]
    true_deoff = [-0.46291038, -0.4420127]
    true_vz = [0.86448656,  0.97591289]

    for meas, truth in zip(raoffs, true_raoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(deoffs, true_deoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(vzs, true_vz):
        assert truth == pytest.approx(meas, abs=threshold)


def test_orbit_e99():
    """
    Test a highly eccentric orbit (ecc=0.99). Again validate against James Graham's orbit code
    """
    # sma, ecc, tau, argp, lan, inc, plx, mtot
    orbital_params = (10, 0.99, 0.3, 0.5, 1.5, 3, 50, 1.5)
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, orbital_params[0], orbital_params[1], orbital_params[2], orbital_params[3],
                                        orbital_params[4], orbital_params[5], orbital_params[6], orbital_params[7])

    true_raoff = [-0.58945575, -0.57148432]
    true_deoff = [-0.44732217, -0.43768456]
    true_vz = [0.39208876,  0.42041953]

    for meas, truth in zip(raoffs, true_raoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(deoffs, true_deoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(vzs, true_vz):
        assert truth == pytest.approx(meas, abs=threshold)

def test_orbit_with_mass():
    """
    Test a orbit where we specify the mass of the body too. This will change the radial velocity, which normally assumes the body is a test particle

    We will test two equal mass bodies, which will reduce the RV signal by 2, comapred to the RV signal of a massless particle in a system with the
    same total mass.
    """
    # sma, ecc, tau, argp, lan, inc, plx, mtot
    orbital_params = (10, 0.99, 0.3, 0.5, 1.5, 3, 50, 1.5)
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, orbital_params[0], orbital_params[1], orbital_params[2], orbital_params[3],
                                        orbital_params[4], orbital_params[5], orbital_params[6], orbital_params[7], mass=orbital_params[7]/2)

    true_raoff = [-0.58945575, -0.57148432]
    true_deoff = [-0.44732217, -0.43768456]
    true_vz = [0.39208876/2,  0.42041953/2]

    for meas, truth in zip(raoffs, true_raoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(deoffs, true_deoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(vzs, true_vz):
        assert truth == pytest.approx(meas, abs=threshold)

if __name__ == "__main__":
    test_orbit_e03()
