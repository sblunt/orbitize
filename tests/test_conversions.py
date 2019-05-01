"""
Test the orbitize.basis which converts orbital elements
"""
import pytest
import numpy as np
import orbitize.basis as basis

def test_tau_t0_conversion():
    """
    Test conversion back and forth
    """
    tau = 0.1
    ref_epoch = 51000 # MJD
    period = 10 # years

    t0 = basis.tau_to_t0(tau, ref_epoch, period)
    assert t0 == pytest.approx(51000 + 365.25, rel=1e-7)

    tau2 = basis.t0_to_tau(t0, ref_epoch, period)
    assert tau == pytest.approx(tau2, rel=1e-7)

    t0 = basis.tau_to_t0(tau, ref_epoch, period, after_date=47000)
    assert t0 == pytest.approx(51000 - 9 * 365.25, rel=1e-7)

    tau3 = basis.t0_to_tau(t0, ref_epoch, period)
    assert tau == pytest.approx(tau3, rel=1e-7)

def test_tau_t0_conversion_vector():
    """
    Make sure it works vectorized.
    """
    taus = np.array([0.1, 0.2])
    ref_epoch = 55000 # MJD
    period = np.array([1, 0.5]) # years

    t0s = basis.tau_to_t0(taus, ref_epoch, period)
    for t0 in t0s:
        assert t0 == pytest.approx(55000 + 365.25/10, rel=1e-7)

def test_switch_tau_basis():
    """
    Switch reference epochs
    """
    old_taus = np.array([0.5, 0.5])
    ref_epoch = np.array([50000, 55000])
    period = np.array([2, 2])
    new_epoch = np.array([50000 + 365.25, 55000 + 365.25])

    new_taus = basis.switch_tau_epoch(old_taus, ref_epoch, new_epoch, period)

    assert new_taus[0] == pytest.approx(0, rel=1e-7)
    assert new_taus[1] == pytest.approx(0, rel=1e-7)

if __name__ == "__main__":
    test_tau_t0_conversion()
    test_tau_t0_conversion_vector()
    test_switch_tau_basis()