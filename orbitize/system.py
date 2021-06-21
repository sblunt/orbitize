import numpy as np
from orbitize import priors, read_input, kepler, conversions# , hipparcos
import astropy.units as u
import astropy.constants as consts

class System(object):
    """
    A class to store information about a system (data & priors)
    and calculate model predictions given a set of orbital
    parameters.

    Args:
        num_secondary_bodies (int): number of secondary bodies in the system.
            Should be at least 1.
        data_table (astropy.table.Table): output from ``orbitize.read_input.read_file()``
        stellar_mass (float): mean mass of the primary, in M_sol. See `fit_secondary_mass`
            docstring below.
        plx (float): mean parallax of the system, in mas
        mass_err (float, optional): uncertainty on ``stellar_mass``, in M_sol
        plx_err (float, optional): uncertainty on ``plx``, in mas
        restrict_angle_ranges (bool, optional): if True, restrict the ranges
            of the position angle of nodes to [0,180)
            to get rid of symmetric double-peaks for imaging-only datasets.
        tau_ref_epoch (float, optional): reference epoch for defining tau (MJD).
            Default is 58849 (Jan 1, 2020).
        fit_secondary_mass (bool, optional): if True, include the dynamical
            mass of the orbiting body as a fitted parameter. If this is set to False, ``stellar_mass``
            is taken to be the total mass of the system. (default: False)
        results (list of orbitize.results.Results): results from an orbit-fit
            will be appended to this list as a Results class.

    Users should initialize an instance of this class, then overwrite
    priors they wish to customize.

    Priors are initialized as a list of ``orbitize.priors.Prior`` objects,
    in the following order::

        semimajor axis 1, eccentricity 1, inclination 1,
        argument of periastron 1, position angle of nodes 1,
        epoch of periastron passage 1,
        [semimajor axis 2, eccentricity 2, etc.],
        [parallax, [mass1, mass2, ..], total_mass/m0]

    where 1 corresponds to the first orbiting object, 2 corresponds
    to the second, etc. Mass1, mass2, ... correspond to masses of secondary
    bodies. If `fit_secondary_mass` is set to True, the last element of this
    list is initialized to the mass of the primary. If not, it is
    initialized to the total system mass.

    Written: Sarah Blunt, Henry Ngo, Jason Wang, 2018
    """

    def __init__(self, num_secondary_bodies, data_table, stellar_mass,
                 plx, mass_err=0, plx_err=0, restrict_angle_ranges=None,
                 tau_ref_epoch=58849, fit_secondary_mass=False, results=None,
                 hipparcos_number=None, fitting_basis='standard', hipparcos_filename=None):

        self.num_secondary_bodies = num_secondary_bodies
        self.sys_priors = []
        self.labels = []
        self.results = []
        self.fit_secondary_mass = fit_secondary_mass
        self.tau_ref_epoch = tau_ref_epoch
        self.restrict_angle_ranges = restrict_angle_ranges
        self.fitting_basis = fitting_basis

        #
        # Group the data in some useful ways
        #

        self.data_table = data_table
        # Creates a copy of the input in case data_table needs to be modified
        self.input_table = self.data_table.copy()

        # Rob: check if instrument column is other than default. If other than default, then separate data table into n number of instruments.
        # gather list of instrument: list_instr = self.data_table['instruments'][name of instrument]
        # List of arrays of indices corresponding to each body

        # instruments = np.unique(self.data_table['instruments']) gives a list of unique names

        self.body_indices = []

        # List of arrays of indices corresponding to epochs in RA/Dec for each body
        self.radec = []

        # List of arrays of indices corresponding to epochs in SEP/PA for each body
        self.seppa = []

        # List of index arrays corresponding to each rv for each body
        self.rv = []

        # instr1_tbl = np.where(self.data_table['instruments'] == list_instr[0])
        # loop through the indices per input_table:
        # rv_indices = np.where(instr1_tbl['quant_type'] == 'rv')
        # ... return the parameter labels for each table
        # ...

        self.fit_astrometry=True
        radec_indices = np.where(self.data_table['quant_type'] == 'radec')
        seppa_indices = np.where(self.data_table['quant_type'] == 'seppa')

        if len(radec_indices[0])==0 and len(seppa_indices[0])==0:
            self.fit_astrometry=False
        rv_indices = np.where(self.data_table['quant_type'] == 'rv')

        # defining all indices to loop through the unique rv instruments to get different offsets and jitters
        instrument_list = np.unique(self.data_table['instrument'])
        inst_indices_all = []
        for inst in instrument_list:
            inst_indices = np.where(self.data_table['instrument'] == inst)
            inst_indices_all.append(inst_indices)

        # defining indices for unique instruments in the data table
        self.rv_instruments = np.unique(self.data_table['instrument'][rv_indices])
        self.rv_inst_indices = []
        for inst in self.rv_instruments:
            inst_indices = np.where(self.data_table['instrument'] == inst)
            self.rv_inst_indices.append(inst_indices)

        # astrometry instruments same for radec and seppa:
        self.astr_instruments = np.unique(
            self.data_table['instrument'][np.where(self.data_table['quant_type'] != 'rv')])
        # save indicies for all of the ra/dec, sep/pa measurements for convenience
        self.all_radec = radec_indices
        self.all_seppa = seppa_indices

        for body_num in np.arange(self.num_secondary_bodies+1):

            self.body_indices.append(
                np.where(self.data_table['object'] == body_num)
            )

            self.radec.append(
                np.intersect1d(self.body_indices[body_num], radec_indices)
            )
            self.seppa.append(
                np.intersect1d(self.body_indices[body_num], seppa_indices)
            )
            self.rv.append(
                np.intersect1d(self.body_indices[body_num], rv_indices)
            )

        # we should track the influence of the planet(s) on each other/the star if we are not fitting massless planets and 
        # we are not fitting relative astrometry of just a single body
        self.track_planet_perturbs = self.fit_secondary_mass and \
                                     ((len(self.radec[1]) + len(self.seppa[1]) + len(self.rv[1]) < len(data_table)) or \
                                      (self.num_secondary_bodies > 1))

        if restrict_angle_ranges:
            angle_upperlim = np.pi
        else:
            angle_upperlim = 2.*np.pi

        #
        # Set priors for each orbital element
        #

        if fitting_basis == 'standard':
            self.best_epochs = None
            for body in np.arange(num_secondary_bodies):
                # Add semimajor axis prior
                self.sys_priors.append(priors.LogUniformPrior(0.001, 1e7))
                self.labels.append('sma{}'.format(body+1))

                # Add eccentricity prior
                self.sys_priors.append(priors.UniformPrior(0., 1.))
                self.labels.append('ecc{}'.format(body+1))

                # Add inclination angle prior
                self.sys_priors.append(priors.SinPrior())
                # self.sys_priors.append(priors.UniformPrior(0., np.pi))# TEST TO COMPARE, CHANGE LATER
                self.labels.append('inc{}'.format(body+1))

                # Add argument of periastron prior
                self.sys_priors.append(priors.UniformPrior(0., 2.*np.pi))
                self.labels.append('aop{}'.format(body+1))

                # Add position angle of nodes prior
                self.sys_priors.append(priors.UniformPrior(0., angle_upperlim))
                self.labels.append('pan{}'.format(body+1))

                # Add epoch of periastron prior.
                self.sys_priors.append(priors.UniformPrior(0., 1.))
                self.labels.append('tau{}'.format(body+1))

        elif fitting_basis == 'XYZ':
            epochs = self.data_table['epoch']
            # Get epochs with least uncertainty, as is done in sampler.py
            convert_warning_print = False
            for body_num in np.arange(self.num_secondary_bodies) + 1:
                if len(self.radec[body_num]) > 0:
                    # only print the warning once. 
                    if not convert_warning_print:
                        print('Converting ra/dec data points in data_table to sep/pa. Original data are stored in input_table.')
                        convert_warning_print = True
                    self.convert_data_table_radec2seppa(body_num=body_num)

            sep_err = self.data_table[np.where(
                self.data_table['quant_type'] == 'seppa')]['quant1_err'].copy()
            meas_object = self.data_table[np.where(
                self.data_table['quant_type'] == 'seppa')]['object'].copy()

            self.best_epochs = []
            self.best_epoch_idx = []
            min_sep_indices = np.argsort(sep_err) # indices of sep err sorted from smallest to higheset
            min_sep_indices_body = meas_object[min_sep_indices] # the corresponding body_num that these sorted measurements correspond to
            for i in range(self.num_secondary_bodies):
                body_num = i + 1
                this_object_meas = np.where(min_sep_indices_body == body_num)[0]
                if np.size(this_object_meas) == 0:
                    # no data, no scaling
                    self.best_epochs.append(None)
                    continue
                # get the smallest measurement belonging to this body
                this_best_epoch_idx = min_sep_indices[this_object_meas][0] # already sorted by argsort
                self.best_epoch_idx.append(this_best_epoch_idx)
                this_best_epoch = epochs[this_best_epoch_idx]
                self.best_epochs.append(this_best_epoch)

            for body in np.arange(num_secondary_bodies):
                # Get the epoch with the least uncertainty for this body
                # curr_idx = self.body_indices[body_num]
                # radec_uncerts = self.data_table['quant1_err'][curr_idx] + self.data_table['quant2_err'][curr_idx]
                # min_uncert = np.where(radec_uncerts == np.amin(radec_uncerts))[0]
                # best_idx = curr_idx[0][min_uncert[0]]
                datapoints_to_take = 3
                best_idx = self.best_epoch_idx[body]
                best_epochs = epochs[best_idx:(best_idx+datapoints_to_take)] # 0 is best, the others are for fitting velocity

                # Get data near best epoch ASSUMING THE BEST IS NOT ONE OF THE LAST TWO EPOCHS OF A GIVEN BODY,
                # also assuming this is in radec
                best_ras = self.input_table['quant1'][best_idx:(best_idx+datapoints_to_take)].copy()
                best_ras_err = self.input_table['quant1_err'][best_idx:(best_idx+datapoints_to_take)].copy()
                best_decs =self.input_table['quant2'][best_idx:(best_idx+datapoints_to_take)].copy()
                best_decs_err = self.input_table['quant2_err'][best_idx:(best_idx+datapoints_to_take)].copy()

                # Convert to AU for prior limits
                best_xs = best_ras / plx 
                best_ys = best_decs / plx 
                best_xs_err = np.sqrt((best_ras_err / best_ras)**2 + (plx_err / plx)**2)*np.absolute(best_xs)
                best_ys_err = np.sqrt((best_decs_err / best_decs)**2 + (plx_err / plx)**2)*np.absolute(best_ys)

                # Least-squares fit on velocity for prior limits
                A = np.vander(best_epochs, 2)

                ATA_x = np.dot(A.T, A / (best_xs_err ** 2)[:, None])
                cov_x = np.linalg.inv(ATA_x)
                w_x = np.linalg.solve(ATA_x, np.dot(A.T, best_xs / best_xs_err ** 2))

                ATA_y = np.dot(A.T, A / (best_ys_err ** 2)[:, None])
                cov_y = np.linalg.inv(ATA_y)
                w_y = np.linalg.solve(ATA_y, np.dot(A.T, best_ys / best_ys_err ** 2))

                x_vel = w_x[0]
                x_vel_err = np.sqrt(cov_x[0, 0])
                y_vel = w_y[0]
                y_vel_err = np.sqrt(cov_y[0, 0])

                x_vel = (( x_vel* u.AU / u.day).to(u.km / u.s)).value
                x_vel_err = ((x_vel_err * u.AU / u.day).to(u.km / u.s)).value
                y_vel = ((y_vel * u.AU / u.day).to(u.km / u.s)).value
                y_vel_err = ((y_vel_err * u.AU / u.day).to(u.km / u.s)).value

                # Propose bounds on absolute Z and Z dot given the energy equation
                mu = consts.G * stellar_mass * u.Msun

                mu_vel = 2 * mu / ((x_vel**2 + y_vel**2) * (u.km / u.s * u.km / u.s))
                z_bound = (np.sqrt(mu_vel**2 - (best_xs[0]**2 + best_ys[0]**2)*u.AU *u.AU)).to(u.AU)
                z_bound = z_bound.value

                mu_pos = 2 * mu / np.sqrt((best_xs[0]**2 + best_ys[0]**2) * (u.AU *u.AU))
                z_vel_bound = (np.sqrt(mu_pos - (x_vel**2 + y_vel**2)*(u.km / u.s * u.km / u.s))).to(u.km / u.s)
                z_vel_bound = z_vel_bound.value

                # Add x-coordinate prior
                num_uncerts_x = 5
                self.sys_priors.append(priors.UniformPrior(best_xs[0] - num_uncerts_x*best_xs_err[0], best_xs[0] + num_uncerts_x*best_xs_err[0]))
                self.labels.append('x{}'.format(body+1))
                
                # Add y-coordinate prior
                num_uncerts_y = 5
                self.sys_priors.append(priors.UniformPrior(best_ys[0] - num_uncerts_y*best_ys_err[0], best_ys[0] + num_uncerts_y*best_ys_err[0]))
                self.labels.append('y{}'.format(body+1))

                # Add z-coordinate prior
                # self.sys_priors.append(priors.UniformPrior(-z_bound,z_bound))
                # self.sys_priors.append(priors.LogUniformPrior(0.0001,z_bound))
                self.sys_priors.append(priors.GaussianPrior(0.,z_bound / 4, no_negatives=False))
                self.labels.append('z{}'.format(body+1))

                # Add x-velocity prior
                num_uncerts_xvel = 5
                self.sys_priors.append(priors.UniformPrior(x_vel - num_uncerts_xvel*x_vel_err, x_vel + num_uncerts_xvel*x_vel_err))
                self.labels.append('xdot{}'.format(body+1))

                # Add y-velocity prior
                num_uncerts_yvel = 5
                self.sys_priors.append(priors.UniformPrior(y_vel - num_uncerts_yvel*y_vel_err, y_vel + num_uncerts_yvel*y_vel_err))
                self.labels.append('ydot{}'.format(body+1))

                # Add z-velocity prior
                # self.sys_priors.append(priors.UniformPrior(-z_vel_bound,z_vel_bound))
                # self.sys_priors.append(priors.LogUniformPrior(0.0001,z_vel_bound))
                self.sys_priors.append(priors.GaussianPrior(0.,z_vel_bound / 4, no_negatives=False))
                self.labels.append('zdot{}'.format(body+1))

        #
        # Set priors on total mass and parallax
        #
        self.labels.append('plx')
        if plx_err > 0:
            self.sys_priors.append(priors.GaussianPrior(plx, plx_err))
        else:
            self.sys_priors.append(plx)

        # if hipparcos_IAD is not None:
        #     self.hipparcos_IAD = hipparcos.HipparcosLogProb(hipparcos_filename, hipparcos_number)

        #     # for now, set broad uniform priors on astrometric params relevant for Hipparcos
        #     self.sys_priors.append(priors.UniformPrior(
        #         self.hipparcos_IAD.pm_ra0 - 10 * self.hipparcos_IAD.pm_ra0_err,
        #         self.hipparcos_IAD.pm_ra0 + 10 * self.hipparcos_IAD.pm_ra0_err)
        #     )
        #     self.labels.append('pm_ra')

        #     self.sys_priors.append(priors.UniformPrior(
        #         self.hipparcos_IAD.pm_dec0 - 10 * self.hipparcos_IAD.pm_dec0_err,
        #         self.hipparcos_IAD.pm_dec0 + 10 * self.hipparcos_IAD.pm_dec0_err)
        #     )
        #     self.labels.append('pm_dec')

        #     self.sys_priors.append(priors.UniformPrior(
        #         - 10 * self.hipparcos_IAD.alpha0_err,
        #         10 * self.hipparcos_IAD.alpha0_err)
        #     )
        #     self.labels.append('alpha0')

        #     self.sys_priors.append(priors.UniformPrior(
        #         - 10 * self.hipparcos_IAD.delta0_err,
        #         10 * self.hipparcos_IAD.delta0_err)
        #     )
        #     self.labels.append('delta0')

        # checking for rv data to include appropriate rv priors:
        if len(self.rv[0]) > 0 and self.fit_secondary_mass:
            # Rob and Lea:
            # for instrument in rv_instruments:
                # add gamma and sigma for each and label each unique gamma and sigma per instrument name (gamma+instrument1, ...)
            for instrument in self.rv_instruments:
                self.sys_priors.append(priors.UniformPrior(-5, 5))  # gamma prior in km/s
                self.labels.append('gamma_{}'.format(instrument))

                self.sys_priors.append(priors.LogUniformPrior(1e-4, 0.05))  # jitter prior in km/s
                self.labels.append('sigma_{}'.format(instrument))

        if self.fit_secondary_mass:
            for body in np.arange(num_secondary_bodies)+1:
                self.sys_priors.append(priors.LogUniformPrior(1e-6, 2))  # in Solar masses for now
                self.labels.append('m{}'.format(body))
            self.labels.append('m0')
        else:
            self.labels.append('mtot')

        # still need to append m0/mtot, even though labels are appended above
        if mass_err > 0:
            self.sys_priors.append(priors.GaussianPrior(stellar_mass, mass_err))
        else:
            self.sys_priors.append(stellar_mass)

        # add labels dictionary for parameter indexing
        self.param_idx = dict(zip(self.labels, np.arange(len(self.labels))))

    def compute_all_orbits(self, params_arr):
        """
        Calls orbitize.kepler.calc_orbit and optionally accounts for multi-body
        interactions, as well as computes total quantities like RV (without jitter/gamma)

        Args:
            params_arr (np.array of float): RxM array
                of fitting parameters, where R is the number of
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in ``System()`` above. If M=1, this can be a 1d array.
        
        Returns:
            tuple of:
                raoff (np.array of float): N_epochs x N_bodies x N_orbits array of
                    RA offsets from barycenter at each epoch.
                decoff (np.array of float): N_epochs x N_bodies x N_orbits array of
                    Dec offsets from barycenter at each epoch.
                vz (np.array of float): N_epochs x N_bodies x N_orbits array of
                    radial velocities at each epoch.

        """

        epochs = self.data_table['epoch']
        n_epochs = len(epochs)

        if len(params_arr.shape) == 1:
            n_orbits = 1
        else:
            n_orbits = params_arr.shape[1]

        ra_kepler = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits)) # N_epochs x N_bodies x N_orbits
        dec_kepler = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits))

        ra_perturb = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits)) 
        dec_perturb = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits))

        vz = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits))

        # mass/mtot used to compute each Keplerian orbit will be needed later to compute perturbations
        if self.track_planet_perturbs:
            masses = np.zeros((self.num_secondary_bodies + 1, n_orbits))
            mtots = np.zeros((self.num_secondary_bodies + 1, n_orbits))

        total_rv0 = 0

        for body_num in np.arange(self.num_secondary_bodies)+1:

            startindex = 6 * (body_num - 1)
            if self.fitting_basis == 'standard':

                sma = params_arr[startindex]
                ecc = params_arr[startindex + 1]
                inc = params_arr[startindex + 2]
                argp = params_arr[startindex + 3]
                lan = params_arr[startindex + 4]
                tau = params_arr[startindex + 5]
            
            elif self.fitting_basis == 'XYZ':
                
                # curr_idx = self.body_indices[body_num]
                # radec_uncerts = self.data_table['quant1_err'][curr_idx] + self.data_table['quant2_err'][curr_idx]
                # min_uncert = np.where(radec_uncerts == np.amin(radec_uncerts))
                best_idx = self.best_epoch_idx[body_num-1]
                constrained_epoch = epochs[best_idx]

                to_convert = np.array([*params_arr[startindex:(startindex+6)],params_arr[6 * self.num_secondary_bodies],params_arr[-1]])
                standard_params = conversions.xyz_to_standard(constrained_epoch, to_convert)

                sma = standard_params[0] 
                ecc = standard_params[1]
                inc = standard_params[2]
                argp = standard_params[3]
                lan = standard_params[4]
                tau = standard_params[5]

            plx = params_arr[6 * self.num_secondary_bodies]

            if self.fit_secondary_mass:

                # mass of secondary bodies are in order from -1-num_bodies until -2 in order.
                mass = params_arr[-1-self.num_secondary_bodies+(body_num-1)]
                m0 = params_arr[-1]

                # For what mtot to use to calculate central potential, we should use the mass enclosed in a sphere with r <= distance of planet. 
                # We need to select all planets with sma < this planet. 
                all_smas = params_arr[0:6*self.num_secondary_bodies:6]
                within_orbit = np.where(all_smas <= sma)
                outside_orbit = np.where(all_smas > sma)
                all_pl_masses = params_arr[-1-self.num_secondary_bodies:-1]
                inside_masses = all_pl_masses[within_orbit]
                mtot = np.sum(inside_masses) + m0

            else:
                # if not fitting for secondary mass, then total mass must be stellar mass
                mass = None
                m0 = None
                mtot = params_arr[-1]
            
            if self.track_planet_perturbs:
                masses[body_num] = mass
                mtots[body_num] = mtot

            # solve Kepler's equation
            raoff, decoff, vz_i = kepler.calc_orbit(
                epochs, sma, ecc, inc, argp, lan, tau, plx, mtot,
                mass_for_Kamp=m0, tau_ref_epoch=self.tau_ref_epoch, tau_warning=False
            )

            # raoff, decoff, vz are scalers if the length of epochs is 1
            if len(epochs) == 1:
                raoff = np.array([raoff])
                decoff = np.array([decoff])
                vz_i = np.array([vz_i])

            if n_orbits == 1:
                raoff = raoff.reshape((n_epochs, 1))
                decoff = decoff.reshape((n_epochs, 1))
                vz_i = vz_i.reshape((n_epochs, 1))

            # add Keplerian ra/deoff for this body to storage arrays
            ra_kepler[:, body_num, :] = raoff 
            dec_kepler[:, body_num, :] = decoff
            vz[:, body_num, :] = vz_i

            # vz_i is the ith companion radial velocity
            if self.fit_secondary_mass:
                vz0 = vz_i * -(mass / m0)  # calculating stellar velocity due to ith companion
                total_rv0 = total_rv0 + vz0  # adding stellar velocity and gamma

        # if we are fitting for the mass of the planets, then they will perturb the star
        # add the perturbation on the star due to this planet on the relative astrometry of the planet that was measured
        # We are superimposing the Keplerian orbits, so we can add it linearly, scaled by the mass. 
        # Because we are in Jacobi coordinates, for companions, we only should model the effect of planets interior to it. 
        # (Jacobi coordinates mean that separation for a given companion is measured relative to the barycenter of all interior companions)
        if self.track_planet_perturbs:
            for body_num in np.arange(self.num_secondary_bodies) + 1:
                if body_num > 0:
                    # for companions, only perturb companion orbits at larger SMAs than this one. 
                    # note the +1, since the 0th planet is body_num 1. 
                    startindex = 6 * (body_num - 1)
                    sma = params_arr[startindex]
                    all_smas = params_arr[0:6*self.num_secondary_bodies:6]
                    outside_orbit = np.where(all_smas > sma)

                    which_perturb_bodies = outside_orbit[0] + 1

                else:
                    # for the star, what we are measuring is its position relative to the system barycenter
                    # so we want to account for all of the bodies.  
                    which_perturb_bodies = np.arange(self.num_secondary_bodies+1)

                for other_body_num in which_perturb_bodies:
                    # skip itself since the the 2-body problem is measuring the planet-star separation already
                    if (body_num == other_body_num) | (body_num == 0):
                        continue

                    ## NOTE: we are only handling astrometry right now (TODO: integrate RV into this)
                    ra_perturb[:, other_body_num, :] += (masses[other_body_num]/mtots[other_body_num]) * ra_kepler[:, body_num, :]
                    dec_perturb[:, other_body_num, :] += (masses[body_num]/mtots[body_num]) * dec_kepler[:, body_num, :] 

        raoff = ra_kepler + ra_perturb
        deoff = dec_kepler + dec_perturb
        vz[:, 0, :] = total_rv0
        if self.fitting_basis == 'XYZ' or self.fitting_basis == 'RRdot':
            if ((ecc >= 1.) | (ecc < 0.)):
                # print("bad stuff")
                # print("raoff is ", raoff)
                # print("raoff times inf is ", np.inf*raoff)
                # print("aodmokea", np.inf*raoff, np.inf*deoff, np.inf*vz)
                raoff[:,:,:] = np.inf
                deoff[:,:,:] = np.inf 
                vz[:,:,:] = np.inf
                return raoff, deoff, vz
            else: 
                return raoff, deoff, vz # MULTIPLY TIMES -np.inf, check for shapes
        elif self.fitting_basis == 'standard':
            return raoff, deoff, vz


    def compute_model(self, params_arr):
        """
        Compute model predictions for an array of fitting parameters. 
        Calls the above compute_all_orbits() function, adds jitter/gamma to
        RV measurements, and propagates these predictions to a model array that
        can be subtracted from a data array to compute chi2. 
        
        Args:
            params_arr (np.array of float): RxM array
                of fitting parameters, where R is the number of
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in ``System()`` above. If M=1, this can be a 1d array.

        Returns:
            np.array of float: Nobsx2xM array model predictions. If M=1, this is
            a 2d array, otherwise it is a 3d array.
        """

        raoff, decoff, vz = self.compute_all_orbits(params_arr)

        if len(params_arr.shape) == 1:
            n_orbits = 1
        else:
            n_orbits = params_arr.shape[1]

        n_epochs = len(self.data_table)
        model = np.zeros((n_epochs, 2, n_orbits))
        jitter = np.zeros((n_epochs, 2, n_orbits))
        gamma = np.zeros((n_epochs, 2, n_orbits))

        if len(self.rv[0]) > 0 and self.fit_secondary_mass: 

            # looping through instruments to get the gammas & jitters
            for rv_idx in range(len(self.rv_instruments)):

                jitter[self.rv_inst_indices[rv_idx], 0] = params_arr[ # [km/s]
                    6 * self.num_secondary_bodies+2+2*rv_idx
                ]
                jitter[self.rv_inst_indices[rv_idx], 1] = np.nan


                gamma[self.rv_inst_indices[rv_idx], 0] = params_arr[
                    6 * self.num_secondary_bodies+1+2*rv_idx
                ] 
                gamma[self.rv_inst_indices[rv_idx], 1] = np.nan

        for body_num in np.arange(self.num_secondary_bodies + 1):

            # for the model points that correspond to this planet's orbit, add the model prediction
            # RA/Dec
            if len(self.radec[body_num]) > 0: # (prevent empty array dimension errors)
                model[self.radec[body_num], 0] = raoff[self.radec[body_num], body_num, :]  # N_epochs x N_bodies x N_orbits
                model[self.radec[body_num], 1] = decoff[self.radec[body_num], body_num, :]

            # Sep/PA
            if len(self.seppa[body_num]) > 0:
                sep, pa = radec2seppa(raoff, decoff)

                model[self.seppa[body_num], 0] = sep[self.seppa[body_num], body_num, :]
                model[self.seppa[body_num], 1] = pa[self.seppa[body_num], body_num, :]

            # RV
            if len(self.rv[body_num]) > 0:
                model[self.rv[body_num], 0] = vz[self.rv[body_num], body_num, :]
                model[self.rv[body_num], 1] = np.nan

        if n_orbits == 1:
            model.reshape((n_epochs, 2))
            jitter.reshape((n_epochs, 2))

        if self.fit_secondary_mass:
            return model + gamma, jitter
        else:
            return model, jitter

    def convert_data_table_radec2seppa(self, body_num=1):
        """
        Converts rows of self.data_table given in radec to seppa.
        Note that self.input_table remains unchanged.

        Args:
            body_num (int): which object to convert (1 = first planet)
        """
        for i in self.radec[body_num]:  # Loop through rows where input provided in radec
            # Get ra/dec values
            ra = self.data_table['quant1'][i]
            ra_err = self.data_table['quant1_err'][i]
            dec = self.data_table['quant2'][i]
            dec_err = self.data_table['quant2_err'][i]
            # Convert to sep/PA
            sep, pa = radec2seppa(ra, dec)
            sep_err = 0.5*(ra_err+dec_err)
            pa_err = np.degrees(sep_err/sep)

            # Update data_table
            self.data_table['quant1'][i] = sep
            self.data_table['quant1_err'][i] = sep_err
            self.data_table['quant2'][i] = pa
            self.data_table['quant2_err'][i] = pa_err
            self.data_table['quant_type'][i] = 'seppa'
            # Update self.radec and self.seppa arrays
            self.radec[body_num] = np.delete(
                self.radec[body_num], np.where(self.radec[body_num] == i)[0])
            self.seppa[body_num] = np.append(self.seppa[body_num], i)

    def add_results(self, results):
        """
        Adds an orbitize.results.Results object to the list in system.results

        Args:
            results (orbitize.results.Results object): add this object to list
        """
        self.results.append(results)

    def clear_results(self):
        """
        Removes all stored results
        """
        self.results = []


def radec2seppa(ra, dec, mod180=False):
    """
    Convenience function for converting from
    right ascension/declination to separation/
    position angle.

    Args:
        ra (np.array of float): array of RA values, in mas
        dec (np.array of float): array of Dec values, in mas
        mod180 (Bool): if True, output PA values will be given
            in range [180, 540) (useful for plotting short
            arcs with PAs that cross 360 during observations)
            (default: False)


    Returns:
        tulple of float: (separation [mas], position angle [deg])

    """
    sep = np.sqrt((ra**2) + (dec**2))
    pa = np.degrees(np.arctan2(ra, dec)) % 360.

    if mod180:
        pa[pa < 180] += 360

    return sep, pa

def seppa2radec(sep, pa):
    """
    Convenience function to convert sep/pa to ra/dec

    Args:
        sep (np.array of float): array of separation in mas
        pa (np.array of float): array of position angles in degrees

    Returns:
        tuple: (ra [mas], dec [mas])
    """
    ra = sep * np.sin(np.radians(pa))
    dec = sep * np.cos(np.radians(pa))

    return ra, dec

def transform_errors(x1, x2, x1_err, x2_err, x12_corr, transform_func, nsamps=100000):
    """
    Transform errors and covariances from one basis to another using a Monte Carlo apporach
    Args:
        x1 (float): planet location in first coordinate (e.g., RA, sep) before transformation
        x2 (float): planet location in the second coordinate (e.g., Dec, PA) before transformation)
        x1_err (float): error in x1
        x2_err (float): error in x2
        x12_corr (float): correlation between x1 and x2
        transform_func (function): function that transforms between (x1, x2) and (x1p, x2p) (the transformed coordinates)
                                    The function signature should look like: `x1p, x2p = transform_func(x1, x2)`
        nsamps (int): number of samples to draw more the Monte Carlo approach. More is slower but more accurate. 
    Returns:
        tuple (x1p_err, x2p_err, x12p_corr): the errors and correlations for x1p,x2p (the transformed coordinates)
    """
    # construct covariance matrix from the terms provided
    cov = np.array([[x1_err**2, x1_err*x2_err*x12_corr], [x1_err*x2_err*x12_corr, x2_err**2]])

    samps = np.random.multivariate_normal([x1, x2], cov, size=nsamps)

    x1p, x2p = transform_func(samps[:,0], samps[:, 1])

    x1p_err = np.std(x1p)
    x2p_err = np.std(x2p)
    x12_corr = np.corrcoef([x1p, x2p])[0,1]

    return x1p_err, x2p_err, x12_corr