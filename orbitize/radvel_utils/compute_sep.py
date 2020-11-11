import numpy as np
import pandas as pd

from radvel.basis import Basis
from orbitize.basis import t0_to_tau
from orbitize.kepler import calc_orbit

def compute_sep(
    df, epochs, basis, mtot, mtot_err, plx, plx_err, n_planets=1, pl_num=1
):
    """
    Computes a sky-projected angular separation posterior given a 
    RadVel-computed DataFrame.

    Args:
        df (pd.DataFrame): Radvel-computed posterior (in any orbital basis)
        epochs: 
        basis (str): basis string of input posterior (see 
            radvel.basis.BASIS_NAMES` for the full list of possibilities). 
        mtot:
        mtot_err:
        plx:
        plx_err: 
        n_planets (int): total number of planets in RadVel posterior
        pl_num (int): planet number used in RadVel fits (e.g. a RadVel label of 
            'per1' implies `pl_num` == 1) 

    Example:

        >> df = pd.read_csv('sample_radvel_chains.csv.bz2', index_col=0)
        >> epochs = Time([2022, 2024], format='decimalyear')
        >> seps, df_orb = compute_sep(
               df, epochs, 'per tc secosw sesinw k', 0.82, 0.02, 312.22, 0.47
           )

    Returns:
        tuple of:
            np.array of size n_epochs x len(df): sky-projected angular 
                separations [mas] at each input epoch
            pd.DataFrame: corresponding orbital posterior in orbitize basis
    """

    myBasis = Basis(basis, n_planets)
    df = myBasis.to_synth(df)
    chain_len = len(df)

    # convert RadVel parameters
    per_day = df['per{}'.format(pl_num)].values
    period_yr = per_day / 365.25
    ecc = df['e{}'.format(pl_num)].values
    omega_st_rad = df['w{}'.format(pl_num)].values
    tp_mjd = df['tp{}'.format(pl_num)].values - 2400000.5
    tau = t0_to_tau(tp_mjd, 58849, period_yr)

    # generate 
    mtot = np.random.normal(mtot, mtot_err, size = chain_len)
    sma = (period_yr**2 * mtot)**(1/3)
    omega_pl_rad = omega_st_rad + np.pi
    parallax = np.random.normal(plx, plx_err, size = chain_len)
    cosi = (2. * np.random.random(size = chain_len)) - 1.
    inc = np.arccos(cosi)
    lan = np.random.random(size = chain_len) * 2. * np.pi

    # returns raoff, deoff in mas
    raoff, deoff, _ = calc_orbit(
        epochs.mjd, sma, ecc, inc, 
        omega_pl_rad, lan, tau, 
        parallax, mtot, tau_ref_epoch=58849
    )
    seps = np.sqrt(raoff**2 + deoff**2)

    df_orb = pd.DataFrame(
        np.transpose([sma, ecc, inc, omega_pl_rad, lan, tau, parallax, mtot]), 
        columns=[
            'sma', 'ecc', 'inc_rad', 'omega_pl_rad', 'lan_rad', 'tau_58849', 
            'plx', 'mtot'
        ]
    )

    return seps, df_orb