from orbitize import driver, DATADIR, plot, system
import multiprocessing as mp
import numpy as np
import astropy.units as u, astropy.constants as cst


def test_rv_default_inst(save_plot=False):
    # Initialize Driver to Run MCMC
    filename = "{}/HD4747.csv".format(DATADIR)

    num_secondary_bodies = 1
    system_mass = 0.84  # [Msol]
    plx = 53.18  # [mas]
    mass_err = 0.04  # [Msol]
    plx_err = 0.12  # [mas]

    num_temps = 5
    num_walkers = 50
    num_threads = mp.cpu_count()  # or a different number if you prefer

    my_driver = driver.Driver(
        filename,
        "MCMC",
        num_secondary_bodies,
        system_mass,
        plx,
        mass_err=mass_err,
        plx_err=plx_err,
        system_kwargs={"fit_secondary_mass": True, "tau_ref_epoch": 0},
        mcmc_kwargs={
            "num_temps": num_temps,
            "num_walkers": num_walkers,
            "num_threads": num_threads,
        },
    )

    total_orbits = 100
    burn_steps = 10
    thin = 2

    # Run Quick Sampler
    m = my_driver.sampler
    m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
    epochs = my_driver.system.data_table["epoch"]

    # Set all posterior values equal to ones from Xuan+ 22 (to visually inspect what the plot should look like)
    m.results.post[:, m.results.system.param_idx['m1']] = (67.2 * u.M_jup / u.M_sun).to('').value
    m.results.post[:, m.results.system.param_idx['sma1']] = 10.0
    m.results.post[:, m.results.system.param_idx['ecc1']] = 0.7317
    m.results.post[:, m.results.system.param_idx['inc1']] = np.radians(48.0)
    m.results.post[:, m.results.system.param_idx['aop1']] = np.radians(267.2)
    m.results.post[:, m.results.system.param_idx['pan1']] = np.radians(89.4)
    m.results.post[:, m.results.system.param_idx['tau1']] = ((2462615 - my_driver.system.tau_ref_epoch) / (33.2 * 365.25)) % 1
    m.results.post[:, m.results.system.param_idx['gamma_defrv']] = 0.0


    # Test plotting with single orbit
    _ = m.results.plot_orbits(
        object_to_plot=1,  # Plots orbits for the first (and only) companion
        num_orbits_to_plot=1,  # Plots orbits of this companion
        start_mjd=epochs[3],  # Minimum MJD for colorbar
        rv_time_series=True,
        plot_astrometry_insts=True,
    )

    # Test plotting with multiple orbits
    _ = m.results.plot_orbits(
        object_to_plot=1,  # Plots orbits for the first (and only) companion
        num_orbits_to_plot=10,  # Plots orbits of this companion
        start_mjd=epochs[3],  # Minimum MJD for colorbar
        rv_time_series=True,
        plot_astrometry_insts=True,
    )


    if save_plot:
        import matplotlib.pyplot as plt
        plt.savefig('HD4747_orbit.png')


def test_rv_multiple_inst():
    filename = "{}/HR7672_joint.csv".format(DATADIR)

    num_secondary_bodies = 1
    system_mass = 1.08  # [Msol]
    plx = 56.2  # [mas]
    mass_err = 0.04  # [Msol]
    plx_err = 0.01  # [mas]

    # MCMC parameters
    num_temps = 5
    num_walkers = 30
    num_threads = 2

    my_driver = driver.Driver(
        filename,
        "MCMC",
        num_secondary_bodies,
        system_mass,
        plx,
        mass_err=mass_err,
        plx_err=plx_err,
        system_kwargs={"fit_secondary_mass": True, "tau_ref_epoch": 0},
        mcmc_kwargs={
            "num_temps": num_temps,
            "num_walkers": num_walkers,
            "num_threads": num_threads,
        },
    )

    total_orbits = 500
    burn_steps = 10
    thin = 2

    m = my_driver.sampler
    m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
    epochs = my_driver.system.data_table["epoch"]

    _ = m.results.plot_orbits(
        object_to_plot=1,
        num_orbits_to_plot=1,
        start_mjd=epochs[
            0
        ],  # Minimum MJD for colorbar (here we choose first data epoch)
        rv_time_series=True,
        plot_astrometry_insts=True,
    )

    # Test plotting with multiple orbits
    _ = m.results.plot_orbits(
        object_to_plot=1,
        num_orbits_to_plot=10,
        start_mjd=epochs[
            0
        ],  # Minimum MJD for colorbar (here we choose first data epoch)
        rv_time_series=True,
        plot_astrometry_insts=True,
    )

def test_secondary_rv():
    """
    Make sure plotting works when all RVs are secondary RVs
    """
    filename = "{}/HR7672_joint.csv".format(DATADIR)

    num_secondary_bodies = 1
    system_mass = 1.08  # [Msol]
    plx = 56.2  # [mas]
    mass_err = 0.04  # [Msol]
    plx_err = 0.01  # [mas]

    # MCMC parameters
    num_temps = 5
    num_walkers = 30
    num_threads = 2

    my_driver = driver.Driver(
        filename,
        "MCMC",
        num_secondary_bodies,
        system_mass,
        plx,
        mass_err=mass_err,
        plx_err=plx_err,
        system_kwargs={"fit_secondary_mass": True, "tau_ref_epoch": 0},
        mcmc_kwargs={
            "num_temps": num_temps,
            "num_walkers": num_walkers,
            "num_threads": num_threads,
        },
    )

    # pretend all the RVs are of the secondary
    my_driver.system.data_table["object"] = 1

    total_orbits = 500
    burn_steps = 10
    thin = 2

    m = my_driver.sampler
    m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
    epochs = my_driver.system.data_table["epoch"]

    plot.plot_orbits(
        my_driver.sampler.results,
        object_to_plot=1,
        num_orbits_to_plot=10,
        start_mjd=epochs[0],
        rv_time_series2=True,
        plot_astrometry_insts=True,
    )

    import matplotlib.pyplot as plt
    plt.savefig('foo.png')

def test_secondary_and_primary_rv():
    """
    Make sure plotting works RVs are a mix of primary and secondary
    """
    filename = "{}/HR7672_joint.csv".format(DATADIR)

    num_secondary_bodies = 1
    system_mass = 1.08  # [Msol]
    plx = 56.2  # [mas]
    mass_err = 0.04  # [Msol]
    plx_err = 0.01  # [mas]

    # MCMC parameters
    num_temps = 5
    num_walkers = 30
    num_threads = 2

    my_driver = driver.Driver(
        filename,
        "MCMC",
        num_secondary_bodies,
        system_mass,
        plx,
        mass_err=mass_err,
        plx_err=plx_err,
        system_kwargs={"fit_secondary_mass": True, "tau_ref_epoch": 0},
        mcmc_kwargs={
            "num_temps": num_temps,
            "num_walkers": num_walkers,
            "num_threads": num_threads,
        },
    )

    # pretend all the RVs are of the secondary
    my_driver.system.data_table["object"] = 1
    my_driver.system.data_table["object"][0:4] = 0

    total_orbits = 500
    burn_steps = 10
    thin = 2

    m = my_driver.sampler
    m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
    epochs = my_driver.system.data_table["epoch"]

    plot.plot_orbits(
        my_driver.sampler.results,
        object_to_plot=1,
        num_orbits_to_plot=10,
        start_mjd=epochs[0],
        rv_time_series=True,
        rv_time_series2=True,
        plot_astrometry_insts=True,
    )




if __name__ == "__main__":
    test_rv_default_inst(save_plot=True)
    # test_rv_multiple_inst()
    # test_secondary_rv()
    # test_secondary_and_primary_rv()
