import numpy as np
import astropy.units as u

def tau_to_t0(tau, ref_epoch, period, after_date=None):
    """
    Convert tau (epoch of periastron in fractional orbital period after ref epoch) to
    T0 (date in days, usually MJD, but works with whatever system ref_epoch is given in)

    Args:
        tau (float or np.array): value of tau to convert
        ref_epoch (float or np.array): date (in days, typically MJD) that tau is defined relative to
        period (float or np.array): period (in years) that tau is noralized with
        after_date (float): T0 will be the first periastron after this date. If None, use ref_epoch.

    Returns:
        t0 (float or np.array): corresponding T0 of the taus
    """
    period_days = period * u.year.to(u.day)

    t0 = tau * (period_days) + ref_epoch

    if after_date is not None:
        num_periods = (after_date - t0)/period_days
        num_periods = int(np.ceil(num_periods))
        
        t0 += num_periods * period_days

    return t0

def t0_to_tau(t0, ref_epoch, period):
    """
    Convert T0 to tau

    Args:
        t0 (float or np.array): value to T0 to convert (days, typically MJD)
        ref_epoch (float or np.array): reference epoch (in days) that tau is defined from. Same system as t0 (e.g., MJD)
        period (float or np.array): period (in years) that tau is defined by

    Returns:
        tau (float or np.array): corresponding taus
    """
    tau = (t0 - ref_epoch)/(period * u.year.to(u.day))
    tau %= 1

    return tau

def switch_tau_epoch(old_tau, old_epoch, new_epoch, period):
    """
    Convert tau to another tau that uses a different referench epoch

    Args:
        old_tau (float or np.array): old tau to convert
        old_epoch (float or np.array): old reference epoch (days, typically MJD)
        new_epoch (float or np.array): new reference epoch (days, same system as old_epoch)
        period (float or np.array): orbital period (years)

    Returns:
        new_tau (float or np.array): new taus
    """
    period_days = period * u.year.to(u.day)

    t0 = tau_to_t0(old_tau, old_epoch, period)
    new_tau = t0_to_tau(t0, new_epoch, period)

    return new_tau