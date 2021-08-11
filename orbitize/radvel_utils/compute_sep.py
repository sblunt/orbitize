import numpy as np
import pandas as pd
from astropy import units as u

from radvel.basis import Basis
from radvel.utils import Msini
from orbitize.basis import t0_to_tau
from orbitize.kepler import calc_orbit

def compute_sep(
    df, epochs, basis, m0, m0_err, plx, plx_err, n_planets=1, pl_num=1
):
    """
    Computes a sky-projected angular separation posterior given a 
    RadVel-computed DataFrame.

    Args:
        df (pd.DataFrame): Radvel-computed posterior (in any orbital basis)
        epochs (np.array of astropy.time.Time): epochs at which to compute 
            separations
        basis (str): basis string of input posterior (see 
            radvel.basis.BASIS_NAMES` for the full list of possibilities). 
        m0 (float): median of primary mass distribution (assumed Gaussian).
        m0_err (float): 1sigma error of primary mass distribution 
            (assumed Gaussian).
        plx (float): median of parallax distribution (assumed Gaussian).
        plx_err: 1sigma error of parallax distribution (assumed Gaussian).
        n_planets (int): total number of planets in RadVel posterior
        pl_num (int): planet number used in RadVel fits (e.g. a RadVel label of 
            'per1' implies `pl_num` == 1) 

    Example:

        >> df = pandas.read_csv('sample_radvel_chains.csv.bz2', index_col=0)
        >> epochs = astropy.time.Time([2022, 2024], format='decimalyear')
        >> seps, df_orb = compute_sep(
               df, epochs, 'per tc secosw sesinw k', 0.82, 0.02, 312.22, 0.47
           )

    Returns:
        tuple of:
            np.array of size (len(epochs) x len(df)): sky-projected angular 
                separations [mas] at each input epoch
            pd.DataFrame: corresponding orbital posterior in orbitize basis
    """

    myBasis = Basis(basis, n_planets)
    df = myBasis.to_synth(df)
    chain_len = len(df)
    tau_ref_epoch = 58849

    # convert RadVel posteriors -> orbitize posteriors
    m_st = np.random.normal(m0, m0_err, size = chain_len)
    semiamp = df['k{}'.format(pl_num)].values
    per_day = df['per{}'.format(pl_num)].values
    period_yr = per_day / 365.25
    ecc = df['e{}'.format(pl_num)].values
    msini = (
        Msini(semiamp, per_day, m_st, ecc, Msini_units='Earth') * 
        (u.M_earth / u.M_sun).to('')
    )
    cosi = (2. * np.random.random(size = chain_len)) - 1.
    inc = np.arccos(cosi)
    m_pl = msini / np.sin(inc)
    mtot = m_st + m_pl
    sma = (period_yr**2 * mtot)**(1/3)
    omega_st_rad = df['w{}'.format(pl_num)].values
    omega_pl_rad = omega_st_rad + np.pi
    parallax = np.random.normal(plx, plx_err, size = chain_len)
    lan = np.random.random_sample(size = chain_len) * 2. * np.pi
    tp_mjd = df['tp{}'.format(pl_num)].values - 2400000.5
    tau = t0_to_tau(tp_mjd, tau_ref_epoch, period_yr)

    # compute projected separation in mas
    raoff, deoff, _ = calc_orbit(
        epochs.mjd, sma, ecc, inc, 
        omega_pl_rad, lan, tau, 
        parallax, mtot, tau_ref_epoch=tau_ref_epoch
    )
    seps = np.sqrt(raoff**2 + deoff**2)

    df_orb = pd.DataFrame(
        np.transpose([sma, ecc, inc, omega_pl_rad, lan, tau, parallax, m_st, m_pl]), 
        columns=[
            'sma', 'ecc', 'inc_rad', 'omega_pl_rad', 'lan_rad', 'tau_58849', 
            'plx', 'm_st', 'mp'
        ]
    )

    return seps, df_orb
