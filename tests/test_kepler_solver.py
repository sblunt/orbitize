"""
Test the orbitize.kepler module which solves for the orbits of the planets
"""
import pytest
import numpy as np
import orbitize.kepler as kepler

threshold = 1e-8

def angle_diff(ang1, ang2):
    # Return the difference between two angles
    return np.arctan2(np.sin(ang1 - ang2), np.cos(ang1 - ang2))

def test_analytical_ecc_anom_solver():
    """
    Test orbitize.kepler._calc_ecc_anom() in the analytical solver regime (e > 0.95) by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """
    mean_anoms=np.linspace(0,2.0*np.pi,1000)
    eccs=np.linspace(0.95,0.999999,100)
    for ee in eccs:
        ecc_anoms = kepler._calc_ecc_anom(mean_anoms,ee,tolerance=1e-9)
        calc_mm = (ecc_anoms - ee*np.sin(ecc_anoms)) % (2*np.pi) # plug solutions into Kepler's equation
        for meas, truth in zip(calc_mm, mean_anoms):
            assert angle_diff(meas, truth) == pytest.approx(0.0, abs=1e-8)

def test_iterative_ecc_anom_solver():
    """
    Test orbitize.kepler._calc_ecc_anom() in the iterative solver regime (e < 0.95) by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """
    mean_anoms=np.linspace(0,2.0*np.pi,100)
    eccs=np.linspace(0,0.9499999,100)
    for ee in eccs:
        ecc_anoms = kepler._calc_ecc_anom(mean_anoms,ee,tolerance=1e-9)
        calc_ma = (ecc_anoms - ee*np.sin(ecc_anoms)) % (2*np.pi) # plug solutions into Kepler's equation
        for meas, truth in zip(calc_ma, mean_anoms):
            assert angle_diff(meas, truth) == pytest.approx(0.0, abs=1e-8)

def test_orbit_e03():
    """
    Test orbitize.kepler.calc_orbit() by comparing this code to the output of James Graham's code which has been used in
    many published papers. Note that orbitize currently uses Rob De Rosa's eccentricity solver.

    Pretty standard orbit with ecc = 0.3
    """
    # sma, ecc, tau, argp, lan, inc, plx, mtot
    orbital_params = np.array([10, 0.3, 0.3, 0.5, 1.5, 3, 50, 1.5])
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

def test_orbit_e03_array():
    """
    Test orbitize.kepler.calc_orbit() with a standard orbit with ecc = 0.3 and an array of keplerian input
    """
    # sma, ecc, tau, argp, lan, inc, plx, mtot
    sma = np.array([10,10,10])
    ecc = np.array([0.3,0.3,0.3])
    tau = np.array([0.3,0.3,0.3])
    argp = np.array([0.5,0.5,0.5])
    lan = np.array([1.5,1.5,1.5])
    inc = np.array([3,3,3])
    plx = np.array([50,50,50])
    mtot = np.array([1.5,1.5,1.5])
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, sma, ecc, tau, argp, lan, inc, plx, mtot)

    true_raoff = np.array([[ 0.15286786,  0.18039408],
                           [ 0.15286786,  0.18039408],
                           [ 0.15286786,  0.18039408]])
    true_deoff = np.array([[-0.46291038, -0.4420127],
                           [-0.46291038, -0.4420127],
                           [-0.46291038, -0.4420127]])
    true_vz    = np.array([[0.86448656,  0.97591289],
                           [0.86448656,  0.97591289],
                           [0.86448656,  0.97591289]])
    for ii in range(0,3):
        for meas, truth in zip(raoffs[ii,:], true_raoff[ii,:]):
            assert truth == pytest.approx(meas, abs=threshold)
        for meas, truth in zip(deoffs[ii,:], true_deoff[ii,:]):
            assert truth == pytest.approx(meas, abs=threshold)
        for meas, truth in zip(vzs[ii,:], true_vz[ii,:]):
            assert truth == pytest.approx(meas, abs=threshold)


def test_orbit_e99():
    """
    Test a highly eccentric orbit (ecc=0.99). Again validate against James Graham's orbit code
    """
    # sma, ecc, tau, argp, lan, inc, plx, mtot
    orbital_params = np.array([10, 0.99, 0.3, 0.5, 1.5, 3, 50, 1.5])
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

    We will test two equal mass bodies, which will reduce the RV signal by 2, compared to the RV signal of a massless particle in a system with the
    same total mass.
    """
    # sma, ecc, tau, argp, lan, inc, plx, mtot
    orbital_params = np.array([10, 0.99, 0.3, 0.5, 1.5, 3, 50, 1.5])
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

def test_orbit_with_mass_array():
    """
    Test orbitize.kepler.calc_orbit() with massive particle on a standard orbit with ecc = 0.3 and an array of keplerian input
    """
    # sma, ecc, tau, argp, lan, inc, plx, mtot
    sma = np.array([10,10,10])
    ecc = np.array([0.3,0.3,0.3])
    tau = np.array([0.3,0.3,0.3])
    argp = np.array([0.5,0.5,0.5])
    lan = np.array([1.5,1.5,1.5])
    inc = np.array([3,3,3])
    plx = np.array([50,50,50])
    mtot = np.array([1.5,1.5,1.5])
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, sma, ecc, tau, argp, lan, inc, plx, mtot, mtot[0]/2)

    true_raoff = np.array([[ 0.15286786,  0.18039408],
                           [ 0.15286786,  0.18039408],
                           [ 0.15286786,  0.18039408]])
    true_deoff = np.array([[-0.46291038, -0.4420127],
                           [-0.46291038, -0.4420127],
                           [-0.46291038, -0.4420127]])
    true_vz    = np.array([[0.86448656/2,  0.97591289/2],
                           [0.86448656/2,  0.97591289/2],
                           [0.86448656/2,  0.97591289/2]])
    for ii in range(0,3):
        for meas, truth in zip(raoffs[ii,:], true_raoff[ii,:]):
            assert truth == pytest.approx(meas, abs=threshold)
        for meas, truth in zip(deoffs[ii,:], true_deoff[ii,:]):
            assert truth == pytest.approx(meas, abs=threshold)
        for meas, truth in zip(vzs[ii,:], true_vz[ii,:]):
            assert truth == pytest.approx(meas, abs=threshold)

if __name__ == "__main__":
    test_analytical_ecc_anom_solver()
    test_iterative_ecc_anom_solver()
    test_orbit_e03()
    test_orbit_e03_array()
    test_orbit_e99()
    test_orbit_with_mass()
    test_orbit_with_mass_array()
