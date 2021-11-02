import os
import numpy as np
import pytest
import astropy.table as table
import astropy
import astropy.units as u
import orbitize
import orbitize.read_input as read_input
import orbitize.kepler as kepler
import orbitize.system as system
import orbitize.sampler as sampler
import orbitize.priors as priors

def test_compute_model():
    """
    Tests that the perturbation of a second planet using compute model gives roughly the amplitude we expect. 
    """
    # generate planet b orbital parameters
    b_params = [1, 0, 0, 0, 0, 0]
    tau_ref_epoch = 0
    mass_b = 0.001 # Msun
    m0 = 1 # Msun
    plx = 1 # mas

    # generate planet c orbital parameters
    # at 0.3 au, and starts on the opposite side of the star relative to b
    c_params = [0.3, 0, 0, np.pi, 0, 0]
    mass_c = 0.002 # Msun

    mtot = m0 + mass_b + mass_c

    period_c = np.sqrt(c_params[0]**3/mtot)
    period_b = np.sqrt(b_params[0]**3/mtot)

    epochs = np.linspace(0, period_c*365.25, 5) + tau_ref_epoch # the full period of c, MJD

    ra_model, dec_model, vz_model = kepler.calc_orbit(
        epochs, b_params[0], b_params[1], b_params[2], b_params[3], b_params[4], 
        b_params[5], plx, mtot, tau_ref_epoch=tau_ref_epoch
    )

    # generate some fake measurements just to feed into system.py to test bookkeeping
    # just make a 1 planet fit for now
    t = table.Table([epochs, np.ones(epochs.shape, dtype=int), ra_model, np.zeros(ra_model.shape), dec_model, np.zeros(dec_model.shape)], 
                     names=["epoch", "object" ,"raoff", "raoff_err","decoff","decoff_err"])
    filename = os.path.join(orbitize.DATADIR, "multiplanet_fake_1planettest.csv")
    t.write(filename, overwrite=True)

    # create the orbitize system and generate model predictions using the standard 1 body model for b, and the 2 body model for b and c
    astrom_dat = read_input.read_file(filename)

    sys_1body = system.System(1, astrom_dat, m0, plx, tau_ref_epoch=tau_ref_epoch, fit_secondary_mass=True)
    sys_2body = system.System(2, astrom_dat, m0, plx, tau_ref_epoch=tau_ref_epoch, fit_secondary_mass=True)

    # model predictions for the 1 body case
    # we had put one measurements of planet b in the data table, so compute_model only does planet b measurements
    params = np.append(b_params, [plx, mass_b, m0])
    radec_1body, _ = sys_1body.compute_model(params)
    ra_1body = radec_1body[:, 0]
    dec_1body = radec_1body[:, 1]

    # model predictions for the 2 body case
    # still only generates predictions of b's location, but including the perturbation for c
    params = np.append(b_params, np.append(c_params, [plx, mass_b, mass_c,  m0]))
    radec_2body, _ = sys_2body.compute_model(params)
    ra_2body = radec_2body[:, 0]
    dec_2body = radec_2body[:, 1]

    ra_diff = ra_2body - ra_1body
    dec_diff = dec_2body - dec_1body
    total_diff = np.sqrt(ra_diff**2 + dec_diff**2)

    # the expected influence of c is mass_c/m0 * sma_c * plx in amplitude
    # just test the first value, because of the face on orbit, we should see it immediately. 
    assert total_diff[0] == pytest.approx(mass_c/m0 * c_params[0] * plx, abs=0.01 * mass_c/m0 * b_params[0] * plx)

    # clean up
    os.system('rm {}'.format(filename))


def test_fit_selfconsist():
    """
    Tests that the masses we get from orbitize! are what we expeect. Note that this does not test for correctness.
    """
    # generate planet b orbital parameters
    b_params = [1, 0, 0, 0, 0, 0.5]
    tau_ref_epoch = 0
    mass_b = 0.001 # Msun
    m0 = 1 # Msun
    plx = 1 # mas

    # generate planet c orbital parameters
    # at 0.3 au, and starts on the opposite side of the star relative to b
    c_params = [0.3, 0, 0, np.pi, 0, 0.5]
    mass_c = 0.002 # Msun
        
    mtot_c = m0 + mass_b + mass_c
    mtot_b = m0 + mass_b

    period_b = np.sqrt(b_params[0]**3/mtot_b)
    period_c = np.sqrt(c_params[0]**3/mtot_c)

    epochs = np.linspace(0, period_b*365.25, 20) + tau_ref_epoch # the full period of b, MJD

    # comptue Keplerian orbit of b
    ra_model_b, dec_model_b, vz_model = kepler.calc_orbit(
        epochs, b_params[0], b_params[1], b_params[2], b_params[3], b_params[4], 
        b_params[5], plx, mtot_b, mass_for_Kamp=m0, tau_ref_epoch=tau_ref_epoch
    )

    # comptue Keplerian orbit of c
    ra_model_c, dec_model_c, vz_model_c = kepler.calc_orbit(
        epochs, c_params[0], c_params[1], c_params[2], c_params[3], c_params[4], 
        c_params[5], plx, mtot_c, tau_ref_epoch=tau_ref_epoch
    )

    # perturb b due to c
    ra_model_b_orig = np.copy(ra_model_b)
    dec_model_b_orig = np.copy(dec_model_b)
    # the sign is positive b/c of 2 negative signs cancelling. 
    ra_model_b += mass_c/m0 * ra_model_c
    dec_model_b += mass_c/m0 * dec_model_c

    # # perturb c due to b
    # ra_model_c += mass_b/m0 * ra_model_b_orig
    # dec_model_c += mass_b/m0 * dec_model_b_orig

    # generate some fake measurements to fit to. Make it with b first
    t = table.Table([epochs, np.ones(epochs.shape, dtype=int), ra_model_b, 0.00001 * np.ones(epochs.shape, dtype=int), dec_model_b, 0.00001 * np.ones(epochs.shape, dtype=int)], 
                     names=["epoch", "object" ,"raoff", "raoff_err","decoff","decoff_err"])
    # add c
    for eps, ra, dec in zip(epochs, ra_model_c, dec_model_c):
        t.add_row([eps, 2, ra, 0.000001, dec, 0.000001])

    filename = os.path.join(orbitize.DATADIR, "multiplanet_fake_2planettest.csv")
    t.write(filename, overwrite=True)

    # create the orbitize system and generate model predictions using the standard 1 body model for b, and the 2 body model for b and c
    astrom_dat = read_input.read_file(filename)
    sys = system.System(2, astrom_dat, m0, plx, tau_ref_epoch=tau_ref_epoch, fit_secondary_mass=True)

    # fix most of the orbital parameters to make the dimensionality a bit smaller
    sys.sys_priors[sys.param_idx['ecc1']] = b_params[1]
    sys.sys_priors[sys.param_idx['inc1']] = b_params[2]
    sys.sys_priors[sys.param_idx['aop1']] = b_params[3]
    sys.sys_priors[sys.param_idx['pan1']] = b_params[4]

    sys.sys_priors[sys.param_idx['ecc2']] = c_params[1]
    sys.sys_priors[sys.param_idx['inc2']] = c_params[2]
    sys.sys_priors[sys.param_idx['aop2']] = c_params[3]
    sys.sys_priors[sys.param_idx['pan2']] = c_params[4]

    sys.sys_priors[sys.param_idx['m1']] = priors.LogUniformPrior(mass_b*0.01, mass_b*100)
    sys.sys_priors[sys.param_idx['m2']] = priors.LogUniformPrior(mass_c*0.01, mass_c*100)

    n_walkers = 30
    samp = sampler.MCMC(sys, num_temps=1, num_walkers=n_walkers, num_threads=1)
    # should have 8 parameters
    assert samp.num_params == 6

    # start walkers near the true location for the orbital parameters
    np.random.seed(123)
    # planet b
    samp.curr_pos[:,0] = np.random.normal(b_params[0], 0.01, n_walkers) # sma
    samp.curr_pos[:,1] = np.random.normal(b_params[-1], 0.01, n_walkers) # tau
    # planet c
    samp.curr_pos[:,2] = np.random.normal(c_params[0], 0.01, n_walkers) # sma
    samp.curr_pos[:,3] = np.random.normal(c_params[-1], 0.01, n_walkers) # tau
    # we will make a fairly broad mass starting position
    samp.curr_pos[:,4] = np.random.uniform(mass_b * 0.25, mass_b * 4, n_walkers) 
    samp.curr_pos[:,5] = np.random.uniform(mass_c * 0.25, mass_c * 4, n_walkers) 
    samp.curr_pos[0,4] = mass_b
    samp.curr_pos[0,5] = mass_c

    samp.run_sampler(n_walkers*50, burn_steps=600)

    res = samp.results

    print(np.median(res.post[:,sys.param_idx['m1']]), np.median(res.post[:,sys.param_idx['m2']]))
    assert np.median(res.post[:,sys.param_idx['sma1']]) == pytest.approx(b_params[0], abs=0.01)
    assert np.median(res.post[:,sys.param_idx['sma2']]) == pytest.approx(c_params[0], abs=0.01)
    assert np.median(res.post[:,sys.param_idx['m2']]) == pytest.approx(mass_c, abs=0.5 * mass_c)

    os.system('rm {}'.format(filename))
    

if __name__ == "__main__":
    test_compute_model()
    test_fit_selfconsist()
