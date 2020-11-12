"""
Test the orbitize.kepler module which solves for the orbits of the planets
"""
import pytest
import sys, pstats, cProfile, os
import numpy as np
import orbitize.kepler as kepler

threshold = 1e-5

def angle_diff(ang1, ang2):
    # Return the difference between two angles
    return np.arctan2(np.sin(ang1 - ang2), np.cos(ang1 - ang2))

def test_analytical_ecc_anom_solver(use_c = False):
    """
    Test orbitize.kepler._calc_ecc_anom() in the analytical solver regime (e > 0.95) by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """
    mean_anoms = np.linspace(0,2.0*np.pi,1000)
    eccs = np.linspace(0.95,0.999999,100)
    for ee in eccs:
        ecc_anoms = kepler._calc_ecc_anom(mean_anoms, ee, tolerance=1e-9, use_c=use_c)
        calc_mm = (ecc_anoms - ee*np.sin(ecc_anoms)) % (2*np.pi) # plug solutions into Kepler's equation
        for meas, truth in zip(calc_mm, mean_anoms):
            assert angle_diff(meas, truth) == pytest.approx(0.0, abs=threshold)

def test_iterative_ecc_anom_solver(use_c = False):
    """
    Test orbitize.kepler._calc_ecc_anom() in the iterative solver regime (e < 0.95) by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """
    mean_anoms = np.linspace(0,2.0*np.pi,100)
    eccs = np.linspace(0,0.9499999,100)
    for ee in eccs:
        ecc_anoms = kepler._calc_ecc_anom(mean_anoms, ee, tolerance=1e-9, use_c=use_c)
        calc_ma = (ecc_anoms - ee*np.sin(ecc_anoms)) % (2*np.pi) # plug solutions into Kepler's equation
        for meas, truth in zip(calc_ma, mean_anoms):
            assert angle_diff(meas, truth) == pytest.approx(0.0, abs=threshold)

def test_c_ecc_anom_solver():
    """
    Test the C implementations in orbitize.kepler._calc_ecc_anom() in the iterative and analytical solver regimes by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """
    if kepler.cext:
        test_iterative_ecc_anom_solver(use_c = True)
        test_analytical_ecc_anom_solver(use_c = True)

def test_orbit_e03():
    """
    Test orbitize.kepler.calc_orbit() by comparing this code to the output of James Graham's code which has been used in
    many published papers. Note that orbitize currently uses Rob De Rosa's eccentricity solver.

    Pretty standard orbit with ecc = 0.3
    """
    # sma, ecc, inc, argp, lan, tau, plx, mtot
    orbital_params = np.array([10, 0.3, 3, 0.5, 1.5, 0.3, 50, 1.5])
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, orbital_params[0], orbital_params[1], orbital_params[2], orbital_params[3],
                                        orbital_params[4], orbital_params[5], orbital_params[6], orbital_params[7], tau_ref_epoch=0)

    true_raoff = [152.86786,  180.39408] #mas
    true_deoff = [-462.91038, -442.0127]
    true_vz = [.86448656,  .97591289]

    for meas, truth in zip(raoffs, true_raoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(deoffs, true_deoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(vzs, true_vz):
        assert truth == pytest.approx(meas, abs=1e-8)

def test_orbit_e03_array():
    """
    Test orbitize.kepler.calc_orbit() with a standard orbit with ecc = 0.3 and an array of keplerian input
    """
    # sma, ecc, inc, argp, lan, tau, plx, mtot
    sma = np.array([10,10,10])
    ecc = np.array([0.3,0.3,0.3])
    inc = np.array([3,3,3])
    argp = np.array([0.5,0.5,0.5])
    lan = np.array([1.5,1.5,1.5])
    tau = np.array([0.3,0.3,0.3])
    plx = np.array([50,50,50])
    mtot = np.array([1.5,1.5,1.5])
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, tau_ref_epoch=0)

    true_raoff = np.array([[ 152.86786, 152.86786, 152.86786],
                           [ 180.39408, 180.39408, 180.39408]])
    true_deoff = np.array([[-462.91038,-462.91038,-462.91038],
                           [-442.0127, -442.0127, -442.0127]])
    true_vz    = np.array([[.86448656, .86448656, .86448656],
                           [.97591289, .97591289, .97591289]])

    for ii in range(0,3):
        for meas, truth in zip(raoffs[:, ii], true_raoff[:,ii]):
            assert truth == pytest.approx(meas, abs=threshold)
        for meas, truth in zip(deoffs[:, ii], true_deoff[:, ii]):
            assert truth == pytest.approx(meas, abs=threshold)
        for meas, truth in zip(vzs[:, ii], true_vz[:, ii]):
            assert truth == pytest.approx(meas, abs=1e-8)


def test_orbit_e99():
    """
    Test a highly eccentric orbit (ecc=0.99). Again validate against James Graham's orbit code
    """
    # sma, ecc, inc, argp, lan, tau, plx, mtot
    orbital_params = np.array([10, 0.99, 3, 0.5, 1.5, 0.3, 50, 1.5])
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, orbital_params[0], orbital_params[1], orbital_params[2], orbital_params[3],
                                        orbital_params[4], orbital_params[5], orbital_params[6], orbital_params[7], tau_ref_epoch=0)

    true_raoff = [-589.45575, -571.48432]
    true_deoff = [-447.32217, -437.68456]
    true_vz = [.39208876,  .42041953]

    for meas, truth in zip(raoffs, true_raoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(deoffs, true_deoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(vzs, true_vz):
        assert truth == pytest.approx(meas, abs=1e-8)

def test_orbit_with_mass():
    """
    Test a orbit where we specify the mass of the body too. This will change the radial velocity, which normally assumes the body is a test particle

    We will test two equal mass bodies, which will reduce the RV signal by 2, compared to the RV signal of a massless particle in a system with the
    same total mass.
    """
    # sma, ecc, inc, argp, lan, tau, plx, mtot
    orbital_params = np.array([10, 0.99, 3, 0.5, 1.5, 0.3, 50, 1.5])
    epochs = np.array([1000, 1101.4])
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, orbital_params[0], orbital_params[1], orbital_params[2], orbital_params[3],
                                        orbital_params[4], orbital_params[5], orbital_params[6], orbital_params[7], mass_for_Kamp=orbital_params[7]/2, tau_ref_epoch=0)

    true_raoff = [-589.45575, -571.48432]
    true_deoff = [-447.32217, -437.68456]
    true_vz = [.39208876/2,  .42041953/2]

    for meas, truth in zip(raoffs, true_raoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(deoffs, true_deoff):
        assert truth == pytest.approx(meas, abs=threshold)
    for meas, truth in zip(vzs, true_vz):
        assert truth == pytest.approx(meas, abs=1e-8)

def test_orbit_with_mass_array():
    """
    Test orbitize.kepler.calc_orbit() with massive particle on a standard orbit with ecc = 0.3 and an array of keplerian input
    """
    # sma, ecc, inc, argp, lan, tau, plx, mtot
    sma = np.array([10,10,10])
    ecc = np.array([0.3,0.3,0.3])
    inc = np.array([3,3,3])
    argp = np.array([0.5,0.5,0.5])
    lan = np.array([1.5,1.5,1.5])
    tau = np.array([0.3,0.3,0.3])
    plx = np.array([50,50,50])
    mtot = np.array([1.5,1.5,1.5])
    epochs = np.array([1000, 1101.4])
    mass = mtot/2
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, mass_for_Kamp=mass, tau_ref_epoch=0)


    true_raoff = np.array([[ 152.86786, 152.86786, 152.86786],
                           [ 180.39408, 180.39408, 180.39408]])
    true_deoff = np.array([[-462.91038,-462.91038, -462.91038],
                           [-442.0127, -442.0127, -442.0127]])
    true_vz    = np.array([[ .86448656/2,.86448656/2, .86448656/2],
                           [ .97591289/2,.97591289/2, .97591289/2]])

    for ii in range(0,3):
        for meas, truth in zip(raoffs[:, ii], true_raoff[:, ii]):
            assert truth == pytest.approx(meas, abs=threshold)
        for meas, truth in zip(deoffs[:, ii], true_deoff[:, ii]):
            assert truth == pytest.approx(meas, abs=threshold)
        for meas, truth in zip(vzs[:, ii], true_vz[:, ii]):
            assert truth == pytest.approx(meas, abs=1e-8)

def test_orbit_scalar():
    """
    Test orbitize.kepler.calc_orbit() with scalar values
    """
    sma = 10
    ecc = 0.3
    inc = 3
    argp = 0.5
    lan = 1.5
    tau = 0.3
    plx = 50
    mtot = 1.5
    epochs = 1000
    raoffs, deoffs, vzs = kepler.calc_orbit(epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, tau_ref_epoch=0)

    true_raoff = 152.86786
    true_deoff = -462.91038
    true_vz    = .86448656

    assert true_raoff == pytest.approx(raoffs, abs=threshold)
    assert true_deoff == pytest.approx(deoffs, abs=threshold)
    assert true_vz    == pytest.approx(vzs, abs=1e-8)

def profile_iterative_ecc_anom_solver(n_orbits = 1000, use_c = True):
    """
    Test orbitize.kepler._calc_ecc_anom() in the iterative solver regime (e < 0.95) by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """

    mean_anoms=np.linspace(0, 2.0*np.pi,n_orbits)
    eccs=np.linspace(0,0.9499999, n_orbits)
    for ee in eccs:
        ecc_anoms = kepler._calc_ecc_anom(mean_anoms, ee, tolerance=1e-9, use_c = use_c)

def profile_mikkola_ecc_anom_solver(n_orbits = 1000, use_c = True):
    """
    Test orbitize.kepler._calc_ecc_anom() in the iterative solver regime (e < 0.95) by comparing the mean anomaly computed from
    _calc_ecc_anom() output vs the input mean anomaly
    """
    mean_anoms=np.linspace(0, 2.0*np.pi,n_orbits)
    eccs=np.linspace(.95,0.999999, n_orbits)
    for ee in eccs:
        ecc_anoms = kepler._calc_ecc_anom(mean_anoms, ee, use_c = use_c)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-profile':
        try:
            n_orbits = int(sys.argv[2])
        except:
            n_orbits = 20000

        profile_name = "Profile.prof"

        print("Profiling Newton: C with {} orbits".format(n_orbits**2))
        cProfile.runctx("profile_iterative_ecc_anom_solver(n_orbits = n_orbits, use_c = True)", globals(), locals(), "Profile.prof")
        s = pstats.Stats(profile_name)
        s.strip_dirs().sort_stats("time").print_stats()

        print("Profiling Newton: Python with {} orbits".format(n_orbits**2))
        cProfile.runctx("profile_iterative_ecc_anom_solver(n_orbits = n_orbits, use_c = False)", globals(), locals(), "Profile.prof")
        s = pstats.Stats(profile_name)
        s.strip_dirs().sort_stats("time").print_stats()

        print("Profiling Mikkola: C with {} orbits".format(n_orbits**2))
        cProfile.runctx("profile_mikkola_ecc_anom_solver(n_orbits = n_orbits, use_c = True)", globals(), locals(), "Profile.prof")
        s = pstats.Stats(profile_name)
        s.strip_dirs().sort_stats("time").print_stats()

        print("Profiling Mikkola: Python with {} orbits".format(n_orbits**2))
        cProfile.runctx("profile_mikkola_ecc_anom_solver(n_orbits = n_orbits, use_c = False)", globals(), locals(), "Profile.prof")
        s = pstats.Stats(profile_name)
        s.strip_dirs().sort_stats("time").print_stats()

        os.remove(profile_name)
    else:
        test_analytical_ecc_anom_solver()
        test_iterative_ecc_anom_solver()
        test_c_ecc_anom_solver()
        test_orbit_e03()
        test_orbit_e03_array()
        test_orbit_e99()
        test_orbit_with_mass()
        test_orbit_with_mass_array()
        test_orbit_scalar()
        print("Done!")
