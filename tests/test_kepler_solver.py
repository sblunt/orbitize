"""
Test the orbitize.kepler module which solves for the orbits of the planets
"""
import pytest
import numpy as np
import orbitize.kepler as kepler

# need to calcualte to a 100 microarcsecs. Should improve this in the future.
threshold = 1e-4

def test_analytical_ecc_anom_solver():
    """
    Test orbitize.kepler._calc_ecc_anom() in the analytical solver regime (e > 0.95) by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """
    mean_anoms=np.linspace(0,2.0*np.pi,1000)
    eccs=np.linspace(0.95,0.999999,100) # Solver only works in elliptical orbit regime (e < 1)
    mm, ee = np.meshgrid(mean_anoms,eccs) # vector for every mean_anom, ecc pair
    # Meshgrid created a grid for every mean_anom, ecc pair
    # We want a flattened vector
    mm = mm.flatten()
    ee = ee.flatten()
    ecc_anoms = kepler._calc_ecc_anom(mm,ee,tolerance=1e-9)
    # the solver changes the values of mm to be within 0 to pi
    ind_change = np.where(ecc_anoms > np.pi)
    ecc_anoms[ind_change] = (2.0 * np.pi) - ecc_anoms[ind_change]
    calc_mm = ecc_anoms - ee*np.sin(ecc_anoms) # plug solutions into Kepler's equation
    for meas, truth in zip(calc_mm, mm):
        assert truth == pytest.approx(meas, abs=1e-8) 

def test_iterative_ecc_anom_solver():
    """
    Test orbitize.kepler._calc_ecc_anom() in the iterative solver regime (e < 0.95) by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """
    mean_anoms=np.linspace(0,2.0*np.pi,100)
    eccs=np.linspace(0,0.9499999,100)
    for ee in eccs:
        ecc_anoms = kepler._calc_ecc_anom(mean_anoms,ee,tolerance=1e-9)
        calc_ma = ecc_anoms - ee*np.sin(ecc_anoms) # plug solutions into Kepler's equation
        for meas, truth in zip(calc_ma, mean_anoms):
            assert truth == pytest.approx(meas, abs=1e-8)

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
    test_iterative_ecc_anom_solver()
    test_orbit_e03()
