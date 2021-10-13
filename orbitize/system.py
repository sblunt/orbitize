import numpy as np
from orbitize import nbody, kepler, basis
from astropy import table

class System(object):
    """
    A class to store information about a system (data & priors)
    and calculate model predictions given a set of orbital
    parameters.

    Args:
        num_secondary_bodies (int): number of secondary bodies in the system.
            Should be at least 1.
        data_table (astropy.table.Table): output from 
            ``orbitize.read_input.read_file()``
        stellar_or_system_mass (float): mass of the primary star (if fitting for
            dynamical masses of both components) or total system mass (if
            fitting using relative astrometry only) [M_sol]
        plx (float): mean parallax of the system, in mas
        mass_err (float, optional): uncertainty on ``stellar_or_system_mass``, in M_sol
        plx_err (float, optional): uncertainty on ``plx``, in mas
        restrict_angle_ranges (bool, optional): if True, restrict the ranges
            of the position angle of nodes to [0,180)
            to get rid of symmetric double-peaks for imaging-only datasets.
        tau_ref_epoch (float, optional): reference epoch for defining tau (MJD).
            Default is 58849 (Jan 1, 2020).
        fit_secondary_mass (bool, optional): if True, include the dynamical
            mass of the orbiting body as a fitted parameter. If this is set to 
            False, ``stellar_or_system_mass`` is taken to be the total mass of the system. 
            (default: False)
        hipparcos_IAD (orbitize.hipparcos.HipparcosLogProb): an object 
            containing information & precomputed values relevant to Hipparcos
            IAD fitting. See hipparcos.py for more details.
        gaia (orbitize.gaia.GaiaLogProb): an object 
            containing information & precomputed values relevant to Gaia
            astrometrry fitting. See gaia.py for more details.
        fitting_basis (str): the name of the class corresponding to the fitting 
            basis to be used. See basis.py for a list of implemented fitting bases.
        use_rebound (bool): if True, use an n-body backend solver instead
            of a Keplerian solver.

    Priors are initialized as a list of orbitize.priors.Prior objects and stored
    in the variable ``System.sys_priors``. You should initialize this class, 
    then overwrite priors you wish to customize. You can use the 
    ``System.param_idx`` attribute to figure out which indices correspond to 
    which fitting parameters. See the "changing priors" tutorial for more detail.  

    Written: Sarah Blunt, Henry Ngo, Jason Wang, 2018
    """

    def __init__(self, num_secondary_bodies, data_table, stellar_or_system_mass,
                 plx, mass_err=0, plx_err=0, restrict_angle_ranges=False,
                 tau_ref_epoch=58849, fit_secondary_mass=False,
                 hipparcos_IAD=None, gaia=None, fitting_basis='Standard', use_rebound=False,
                 ):

        self.num_secondary_bodies = num_secondary_bodies
        self.data_table = data_table
        self.stellar_or_system_mass = stellar_or_system_mass
        self.plx = plx
        self.mass_err = mass_err
        self.plx_err = plx_err
        self.restrict_angle_ranges = restrict_angle_ranges
        self.tau_ref_epoch = tau_ref_epoch
        self.fit_secondary_mass = fit_secondary_mass
        self.hipparcos_IAD = hipparcos_IAD
        self.gaia = gaia
        self.fitting_basis = fitting_basis
        self.use_rebound = use_rebound

        self.best_epochs = []
        self.input_table = self.data_table.copy()

        # Group the data in some useful ways

        self.body_indices = []

        # List of arrays of indices corresponding to epochs in RA/Dec for each body
        self.radec = []

        # List of arrays of indices corresponding to epochs in SEP/PA for each body
        self.seppa = []

        # List of index arrays corresponding to each rv for each body
        self.rv = []

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

        # we should track the influence of the planet(s) on each other/the star if:
        # we are not fitting massless planets and 
        # we have more than 1 companion OR we have stellar astrometry
        self.track_planet_perturbs = (
            self.fit_secondary_mass and 
            (
                (len(self.radec[0]) + len(self.seppa[0] > 0) or
                (self.num_secondary_bodies > 1)
                )
            )
        )

        if self.hipparcos_IAD is not None:
            self.track_planet_perturbs = True

        if self.restrict_angle_ranges:
            angle_upperlim = np.pi
        else:
            angle_upperlim = 2.*np.pi

        # Check for rv data
        contains_rv = False
        if len(self.rv[0]) > 0:
            contains_rv = True

        # Assign priors for the given basis set
        self.extra_basis_kwargs = {}
        basis_obj = getattr(basis, self.fitting_basis)

        # Obtain extra necessary data to assign priors for XYZ
        if self.fitting_basis == 'XYZ':
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

            astr_inds = np.where(self.input_table['object'] > 0)[0]
            astr_data = self.input_table[astr_inds]
            epochs = astr_data['epoch']

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

            self.extra_basis_kwargs = {'data_table':astr_data, 'best_epoch_idx':self.best_epoch_idx, 'epochs':epochs}

        self.basis = basis_obj(
            self.stellar_or_system_mass, self.mass_err, self.plx, self.plx_err, self.num_secondary_bodies, 
            self.fit_secondary_mass, angle_upperlim=angle_upperlim, 
            hipparcos_IAD=self.hipparcos_IAD, rv=contains_rv, 
            rv_instruments=self.rv_instruments, **self.extra_basis_kwargs
        )

        self.basis.verify_params()
        self.sys_priors, self.labels = self.basis.construct_priors()

        self.secondary_mass_indx = [
            self.basis.standard_basis_idx[i] for i in self.basis.standard_basis_idx.keys() if (
                i.startswith('m') and
                not i.endswith('0')
            )
        ]
    
        self.sma_indx = [
            self.basis.standard_basis_idx[i] for i in self.basis.standard_basis_idx.keys() if (
                i.startswith('sma')
            )
        ]
        self.ecc_indx = [
            self.basis.standard_basis_idx[i] for i in self.basis.standard_basis_idx.keys() if (
                i.startswith('ecc')
            )
        ]
        self.inc_indx = [
            self.basis.standard_basis_idx[i] for i in self.basis.standard_basis_idx.keys() if (
                i.startswith('inc')
            )
        ]
        self.aop_indx = [
            self.basis.standard_basis_idx[i] for i in self.basis.standard_basis_idx.keys() if (
                i.startswith('aop')
            )
        ]
        self.pan_indx = [
            self.basis.standard_basis_idx[i] for i in self.basis.standard_basis_idx.keys() if (
                i.startswith('pan')
            )
        ]
        self.tau_indx = [
            self.basis.standard_basis_idx[i] for i in self.basis.standard_basis_idx.keys() if (
                i.startswith('tau')
            )
        ]
        self.mpl_idx = [
            self.basis.standard_basis_idx[i] for i in self.basis.standard_basis_idx.keys() if (
                i.startswith('m') and i[1:] not in ['tot', '0']
            )
        ]

        self.param_idx = self.basis.param_idx

    def save(self, hf):
        """
        Saves the current object to an hdf5 file

        Args:
            hf (h5py._hl.files.File): a currently open hdf5 file in which
                to save the object.        
        """

        hf.attrs['num_secondary_bodies'] = self.num_secondary_bodies

        hf.create_dataset('data', data=self.input_table)

        hf.attrs['restrict_angle_ranges'] = self.restrict_angle_ranges
        hf.attrs['tau_ref_epoch'] = self.tau_ref_epoch
        hf.attrs['stellar_or_system_mass'] = self.stellar_or_system_mass
        hf.attrs['plx'] = self.plx
        hf.attrs['mass_err'] = self.mass_err
        hf.attrs['plx_err'] = self.plx_err
        hf.attrs['fit_secondary_mass'] = self.fit_secondary_mass

        if self.gaia is not None:
            self.gaia._save(hf)
        elif self.hipparcos_IAD is not None:
            self.hipparcos_IAD._save(hf)
        hf.attrs['fitting_basis'] = self.fitting_basis
        hf.attrs['use_rebound'] = self.use_rebound

        

    def compute_all_orbits(self, params_arr, epochs=None, comp_rebound=False):
        """
        Calls orbitize.kepler.calc_orbit and optionally accounts for multi-body
        interactions. Also computes total quantities like RV (without jitter/gamma)

        Args:
            params_arr (np.array of float): RxM array
                of fitting parameters, where R is the number of
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in ``System()`` above. If M=1, this can be a 1d array.
            epochs (np.array of float): epochs (in mjd) at which to compute
                orbit predictions.
            comp_rebound (bool, optional): A secondary optional input for 
                use of N-body solver Rebound; by default, this will be set
                to false and a Kepler solver will be used instead. 
        
        Returns:
            tuple:

                raoff (np.array of float): N_epochs x N_bodies x N_orbits array of
                    RA offsets from barycenter at each epoch.

                decoff (np.array of float): N_epochs x N_bodies x N_orbits array of
                    Dec offsets from barycenter at each epoch.
                    
                vz (np.array of float): N_epochs x N_bodies x N_orbits array of
                    radial velocities at each epoch.

        """

        if epochs is None:
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

        if comp_rebound or self.use_rebound:
                
            sma = params_arr[self.sma_indx]
            ecc = params_arr[self.ecc_indx]
            inc = params_arr[self.inc_indx]
            argp = params_arr[self.aop_indx]
            lan = params_arr[self.pan_indx]
            tau = params_arr[self.tau_indx]
            plx = params_arr[self.basis.standard_basis_idx['plx']]

            if self.fit_secondary_mass:
                m_pl = params_arr[self.mpl_idx]
                m0 = params_arr[self.basis.param_idx['m0']]
                mtot = m0 + sum(m_pl)
            else:
                m_pl = np.zeros(self.num_secondary_bodies)
                # if not fitting for secondary mass, then total mass must be stellar mass
                mtot = params_arr[self.basis.param_idx['mtot']]
            
            raoff, deoff, vz = nbody.calc_orbit(epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, tau_ref_epoch=self.tau_ref_epoch, m_pl=m_pl, output_star=True)

        else:
                for body_num in np.arange(self.num_secondary_bodies)+1:

                    sma = params_arr[self.basis.standard_basis_idx['sma{}'.format(body_num)]]
                    ecc = params_arr[self.basis.standard_basis_idx['ecc{}'.format(body_num)]]
                    inc = params_arr[self.basis.standard_basis_idx['inc{}'.format(body_num)]]
                    argp = params_arr[self.basis.standard_basis_idx['aop{}'.format(body_num)]]
                    lan = params_arr[self.basis.standard_basis_idx['pan{}'.format(body_num)]]
                    tau = params_arr[self.basis.standard_basis_idx['tau{}'.format(body_num)]]
                    plx = params_arr[self.basis.standard_basis_idx['plx']]

                    if self.fit_secondary_mass:
                        # mass of secondary bodies are in order from -1-num_bodies until -2 in order.
                        mass = params_arr[self.basis.standard_basis_idx['m{}'.format(body_num)]]
                        m0 = params_arr[self.basis.standard_basis_idx['m0']]

                        # For what mtot to use to calculate central potential, we should use the mass enclosed in a sphere with r <= distance of planet. 
                        # We need to select all planets with sma < this planet. 
                        all_smas = params_arr[self.sma_indx]
                        within_orbit = np.where(all_smas <= sma)
                        outside_orbit = np.where(all_smas > sma)
                        all_pl_masses = params_arr[self.secondary_mass_indx]
                        inside_masses = all_pl_masses[within_orbit]
                        mtot = np.sum(inside_masses) + m0

                    else:
                        m_pl = np.zeros(self.num_secondary_bodies)
                        # if not fitting for secondary mass, then total mass must be stellar mass
                        mass = None
                        m0 = None
                        mtot = params_arr[self.basis.standard_basis_idx['mtot']]
                    
                    if self.track_planet_perturbs:
                        masses[body_num] = mass
                        mtots[body_num] = mtot

                    # solve Kepler's equation
                    raoff, decoff, vz_i = kepler.calc_orbit(
                        epochs, sma, ecc, inc, argp, lan, tau, plx, mtot,
                        mass_for_Kamp=m0, tau_ref_epoch=self.tau_ref_epoch
                    )

                    # raoff, decoff, vz are scalers if the length of epochs is 1
                    if len(epochs) == 1:
                        raoff = np.array([raoff])
                        decoff = np.array([decoff])
                        vz_i = np.array([vz_i])

                    # add Keplerian ra/deoff for this body to storage arrays
                    ra_kepler[:, body_num, :] = np.reshape(raoff, (n_epochs, n_orbits)) 
                    dec_kepler[:, body_num, :] = np.reshape(decoff, (n_epochs, n_orbits)) 
                    vz[:, body_num, :] = np.reshape(vz_i, (n_epochs, n_orbits)) 

                    # vz_i is the ith companion radial velocity
                    if self.fit_secondary_mass:
                        vz0 = np.reshape(vz_i * -(mass / m0), (n_epochs, n_orbits)) # calculating stellar velocity due to ith companion
                        vz[:, 0, :] += vz0  # adding stellar velocity and gamma

                # if we are fitting for the mass of the planets, then they will perturb the star
                # add the perturbation on the star due to this planet on the relative astrometry of the planet that was measured
                # We are superimposing the Keplerian orbits, so we can add it linearly, scaled by the mass. 
                # Because we are in Jacobi coordinates, for companions, we only should model the effect of planets interior to it. 
                # (Jacobi coordinates mean that separation for a given companion is measured relative to the barycenter of all interior companions)
                if self.track_planet_perturbs:
                    for body_num in np.arange(self.num_secondary_bodies + 1):

                        if body_num > 0:
                            # for companions, only perturb companion orbits at larger SMAs than this one. 
                            sma = params_arr[self.basis.standard_basis_idx['sma{}'.format(body_num)]]
                            all_smas = params_arr[self.sma_indx]
                            outside_orbit = np.where(all_smas > sma)[0]
                            which_perturb_bodies = outside_orbit + 1

                            # the planet will also perturb the star
                            which_perturb_bodies = np.append([0], which_perturb_bodies)

                        else:
                            # for the star, what we are measuring is its position relative to the system barycenter
                            # so we want to account for all of the bodies.  
                            which_perturb_bodies = np.arange(self.num_secondary_bodies+1)

                        for other_body_num in which_perturb_bodies:
                            # skip itself since the the 2-body problem is measuring the planet-star separation already
                            if (body_num == other_body_num) | (body_num == 0):
                                continue

                            ## NOTE: we are only handling astrometry right now (TODO: integrate RV into this)
                            # this computes the perturbation on the other body due to the current body

                            # star is perturbed in opposite direction
                            if other_body_num == 0:
                                ra_perturb[:, other_body_num, :] -= (masses[body_num]/mtots[body_num]) * ra_kepler[:, body_num, :]
                                dec_perturb[:, other_body_num, :] -= (masses[body_num]/mtots[body_num]) * dec_kepler[:, body_num, :] 
                            
                            else:
                                ra_perturb[:, other_body_num, :] += (masses[body_num]/mtots[body_num]) * ra_kepler[:, body_num, :]
                                dec_perturb[:, other_body_num, :] += (masses[body_num]/mtots[body_num]) * dec_kepler[:, body_num, :] 

                raoff = ra_kepler + ra_perturb
                deoff = dec_kepler + dec_perturb

        if self.fitting_basis == 'XYZ':
            # Find and filter out unbound orbits
            bad_orbits = np.where(np.logical_or(ecc >= 1., ecc < 0.))[0]
            if (bad_orbits.size != 0):
                raoff[:,:, bad_orbits] = np.inf
                deoff[:,:, bad_orbits] = np.inf 
                vz[:,:, bad_orbits] = np.inf
                return raoff, deoff, vz
            else: 
                return raoff, deoff, vz 
        else:
            return raoff, deoff, vz


    def compute_model(self, params_arr, use_rebound=False):
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
            use_rebound (bool, optional): A secondary optional input for 
                use of N-body solver Rebound; by default, this will be set
                to false and a Kepler solver will be used instead.

        Returns:
            tuple of:
                np.array of float: Nobsx2xM array model predictions. If M=1, this is
                    a 2d array, otherwise it is a 3d array.
                np.array of float: Nobsx2xM array jitter predictions. If M=1, this is
                    a 2d array, otherwise it is a 3d array.
        """

        to_convert = np.copy(params_arr)
        standard_params_arr = self.basis.to_standard_basis(to_convert)      

        if use_rebound:
            raoff, decoff, vz = self.compute_all_orbits(standard_params_arr, comp_rebound=True)
        else:
            raoff, decoff, vz = self.compute_all_orbits(standard_params_arr)

        if len(standard_params_arr.shape) == 1:
            n_orbits = 1
        else:
            n_orbits = standard_params_arr.shape[1]

        n_epochs = len(self.data_table)
        model = np.zeros((n_epochs, 2, n_orbits))
        jitter = np.zeros((n_epochs, 2, n_orbits))
        gamma = np.zeros((n_epochs, 2, n_orbits))

        if len(self.rv[0]) > 0 and self.fit_secondary_mass: 

            # looping through instruments to get the gammas & jitters
            for rv_idx in range(len(self.rv_instruments)):

                jitter[self.rv_inst_indices[rv_idx], 0] = standard_params_arr[ # [km/s]
                    self.basis.standard_basis_idx['sigma_{}'.format(self.rv_instruments[rv_idx])]
                ]
                jitter[self.rv_inst_indices[rv_idx], 1] = np.nan


                gamma[self.rv_inst_indices[rv_idx], 0] = standard_params_arr[
                    self.basis.standard_basis_idx['gamma_{}'.format(self.rv_instruments[rv_idx])]
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
            model = model.reshape((n_epochs, 2))
            jitter = jitter.reshape((n_epochs, 2))
            gamma = gamma.reshape((n_epochs, 2))

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
            radec_corr = self.data_table['quant12_corr'][i]
            # Convert to sep/PA
            sep, pa = radec2seppa(ra, dec)

            if np.isnan(radec_corr): 
                # E-Z
                sep_err = 0.5*(ra_err+dec_err)
                pa_err = np.degrees(sep_err/sep)
                seppa_corr = np.nan
            else:
                sep_err, pa_err, seppa_corr = transform_errors(ra, dec, ra_err, dec_err, radec_corr, radec2seppa)

            # Update data_table
            self.data_table['quant1'][i] = sep
            self.data_table['quant1_err'][i] = sep_err
            self.data_table['quant2'][i] = pa
            self.data_table['quant2_err'][i] = pa_err
            self.data_table['quant12_corr'][i] = seppa_corr
            self.data_table['quant_type'][i] = 'seppa'
            # Update self.radec and self.seppa arrays
            self.radec[body_num] = np.delete(
                self.radec[body_num], np.where(self.radec[body_num] == i)[0])
            self.seppa[body_num] = np.append(self.seppa[body_num], i)


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
        tuple of float: (separation [mas], position angle [deg])

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
    Transform errors and covariances from one basis to another using a Monte Carlo 
    apporach
    
   Args:
        x1 (float): planet location in first coordinate (e.g., RA, sep) before 
            transformation
        x2 (float): planet location in the second coordinate (e.g., Dec, PA) 
            before transformation)
        x1_err (float): error in x1
        x2_err (float): error in x2
        x12_corr (float): correlation between x1 and x2
        transform_func (function): function that transforms between (x1, x2) 
            and (x1p, x2p) (the transformed coordinates). The function signature 
            should look like: `x1p, x2p = transform_func(x1, x2)`
        nsamps (int): number of samples to draw more the Monte Carlo approach. 
            More is slower but more accurate. 
    Returns:
        tuple (x1p_err, x2p_err, x12p_corr): the errors and correlations for 
            x1p,x2p (the transformed coordinates)
    """

    if np.isnan(x12_corr):
        x12_corr = 0.

    # construct covariance matrix from the terms provided
    cov = np.array([[x1_err**2, x1_err*x2_err*x12_corr], [x1_err*x2_err*x12_corr, x2_err**2]])

    samps = np.random.multivariate_normal([x1, x2], cov, size=nsamps)

    x1p, x2p = transform_func(samps[:,0], samps[:, 1])

    x1p_err = np.std(x1p)
    x2p_err = np.std(x2p)
    x12_corr = np.corrcoef([x1p, x2p])[0,1]

    return x1p_err, x2p_err, x12_corr
