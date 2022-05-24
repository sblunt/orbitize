import numpy as np
import astropy.units as u
import astropy.constants as consts
import abc
import time
from astropy.time import Time

import emcee
import ptemcee
import multiprocessing as mp
from multiprocessing import Pool

import orbitize.lnlike
import orbitize.priors
import orbitize.kepler
from orbitize import cuda_ext

import orbitize.results
import matplotlib.pyplot as plt

class Sampler(abc.ABC):
    """
    Abstract base class for sampler objects.
    All sampler objects should inherit from this class.

    Written: Sarah Blunt, 2018
    """

    def __init__(self, system, like='chi2_lnlike', custom_lnlike=None, chi2_type='standard'):
        self.system = system

        # check if `like` is a string or a function
        if callable(like):
            self.lnlike = like
        else:
            self.lnlike = getattr(orbitize.lnlike, like)

        self.custom_lnlike = custom_lnlike
        self.chi2_type = chi2_type
        # check if need to handle covariances
        self.has_corr = np.any(~np.isnan(self.system.data_table['quant12_corr']))

    @abc.abstractmethod
    def run_sampler(self, total_orbits):
        pass

    def _logl(self, params):
        """
        log likelihood function that interfaces with the orbitize objects
        Comptues the sum of the log likelihoods of the data given the input model

        Args:
            params (np.array of float): RxM array
                of fitting parameters, where R is the number of
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in System() above. If M=1, this can be a 1d array.

        Returns:
            float: sum of all log likelihoods of the data given input model

        """
        # compute the model based on system params
        model, jitter = self.system.compute_model(params)

        # fold data/errors to match model output shape. In particualr, quant1/quant2 are interleaved
        data = np.array([self.system.data_table['quant1'], self.system.data_table['quant2']]).T

        # errors below required for lnlike function below
        errs = np.array([self.system.data_table['quant1_err'],
                         self.system.data_table['quant2_err']]).T
        # covariances/correlations, if applicable
        # we're doing this check now because the likelihood computation is much faster if we can skip it.
        if self.has_corr:
            corrs = self.system.data_table['quant12_corr']
        else:
            corrs = None

        # grab all seppa indices
        seppa_indices = self.system.all_seppa

        # compute lnlike
        lnlikes = self.lnlike(data, errs, corrs, model, jitter, seppa_indices, chi2_type=self.chi2_type)

        # return sum of lnlikes (aka product of likeliehoods)
        lnlikes_sum = np.nansum(lnlikes, axis=(0, 1))

        if self.custom_lnlike is not None:
            lnlikes_sum += self.custom_lnlike(params)
        
        if self.system.hipparcos_IAD is not None:

            # compute Ra/Dec predictions at the Hipparcos IAD epochs
            raoff_model, deoff_model, _ = self.system.compute_all_orbits(
                params, epochs=self.system.hipparcos_IAD.epochs_mjd
            ) 

            raoff_model_hip_epoch, deoff_model_hip_epoch, _ = self.system.compute_all_orbits(
                params, epochs=Time([1991.25], format='decimalyear').mjd
            ) 

            # subtract off position of star at reference Hipparcos epoch
            raoff_model[:,0,:] -= raoff_model_hip_epoch[:,0,:]
            deoff_model[:,0,:] -= deoff_model_hip_epoch[:,0,:]

            # select body 0 raoff/deoff predictions & feed into Hip IAD lnlike fn
            lnlikes_sum += self.system.hipparcos_IAD.compute_lnlike(
                raoff_model[:,0,:], deoff_model[:,0,:], params, self.system.param_idx
            )

            if self.system.gaia is not None:

                gaiahip_epochs = Time(
                    [self.system.gaia.hipparcos_epoch, self.system.gaia.gaia_epoch], 
                    format='decimalyear'
                ).mjd

                # compute Ra/Dec predictions at the Gaia epoch
                raoff_model, deoff_model, _ = self.system.compute_all_orbits(
                    params, epochs=gaiahip_epochs
                ) 

                # select body 0 raoff/deoff predictions & feed into Gaia module lnlike fn
                lnlikes_sum += self.system.gaia.compute_lnlike(
                    raoff_model[:,0,:], deoff_model[:,0,:], params, self.system.param_idx
                )

        return lnlikes_sum


class OFTI(Sampler,):
    """
    OFTI Sampler

    Args:
        system (system.System): ``system.System`` object
        like (string): name of likelihood function in ``lnlike.py``
        custom_lnlike (func): ability to include an addition custom likelihood 
            function in the fit. The function looks like 
            ``clnlikes = custon_lnlike(params)`` where ``params`` is a RxM array
            of fitting parameters, where R is the number of orbital paramters 
            (can be passed in system.compute_model()),
            and M is the number of orbits we need model predictions for. 
            It returns ``clnlikes`` which is an array of
            length M, or it can be a single float if M = 1.

    Written: Isabel Angelo, Sarah Blunt, Logan Pearce, 2018
    """
    def __init__(self, system, like='chi2_lnlike', custom_lnlike=None,chi2_type='standard'):

        super(OFTI, self).__init__(system, like=like, chi2_type=chi2_type, custom_lnlike=custom_lnlike)

        if (
            (self.system.hipparcos_IAD is not None) or 
            (len(self.system.rv[0] > 0))
        ):
            raise NotImplementedError(
                """
                You can only use OFTI with relative astrometry measurements 
                (no Hipparcos IAD or RVs... yet). Use MCMC, you overachiever, and
                settle in for a nice long orbit-fit. (But seriously, if you want 
                this functionality, let us know!)
                """
            )

        # compute priors and columns containing ra/dec and sep/pa
        self.priors = self.system.sys_priors

        # convert RA/Dec rows to sep/PA
        convert_warning_print = False
        for body_num in np.arange(self.system.num_secondary_bodies) + 1:
            if len(self.system.radec[body_num]) > 0:
                # only print the warning once.
                if not convert_warning_print:
                    print('Converting ra/dec data points in data_table to sep/pa. Original data are stored in input_table.')
                    convert_warning_print = True
                self.system.convert_data_table_radec2seppa(body_num=body_num)

        # these are of type astropy.table.column
        self.sep_observed = self.system.data_table[np.where(
            self.system.data_table['quant_type'] == 'seppa')]['quant1'].copy()
        self.pa_observed = self.system.data_table[np.where(
            self.system.data_table['quant_type'] == 'seppa')]['quant2'].copy()
        self.sep_err = self.system.data_table[np.where(
            self.system.data_table['quant_type'] == 'seppa')]['quant1_err'].copy()
        self.pa_err = self.system.data_table[np.where(
            self.system.data_table['quant_type'] == 'seppa')]['quant2_err'].copy()
        self.meas_object = self.system.data_table[np.where(
            self.system.data_table['quant_type'] == 'seppa')]['object'].copy()

        # this is OK, ONLY IF we are only using self.epochs for computing RA/Dec from Keplerian elements
        self.epochs = np.array(self.system.data_table['epoch']) - self.system.tau_ref_epoch

        # distinguishing all epochs from sep/pa epochs
        self.epochs_seppa = np.array(self.system.data_table[np.where(
            self.system.data_table['quant_type'] == 'seppa')]['epoch']) - self.system.tau_ref_epoch

        self.epochs_rv = np.array(self.system.data_table[np.where(
            self.system.data_table['quant_type'] == 'rv')]['epoch']) - self.system.tau_ref_epoch

        # choose scale-and-rotate epoch
        # for multiplanet support, this is now a list.
        # For each planet, we find the measurment for it that corresponds to the smallest astrometric error
        self.epoch_idx = []
        min_sep_indices = np.argsort(self.sep_err) # indices of sep err sorted from smallest to higheset
        min_sep_indices_body = self.meas_object[min_sep_indices] # the corresponding body_num that these sorted measurements correspond to
        for i in range(self.system.num_secondary_bodies):
            body_num = i + 1
            this_object_meas = np.where(min_sep_indices_body == body_num)[0]
            if np.size(this_object_meas) == 0:
                # no data, no scaling
                self.epoch_idx.append(None)
                continue
            # get the smallest measurement belonging to this body
            best_epoch = min_sep_indices[this_object_meas][0] # already sorted by argsort
            self.epoch_idx.append(best_epoch)

        if len(self.system.rv[0]) > 0 and self.system.fit_secondary_mass:  # checking for RV data
            self.rv_observed = self.system.data_table[np.where(
                self.system.data_table['quant_type'] == 'rv')]['quant1'].copy()
            self.rv_err = self.system.data_table[np.where(
                self.system.data_table['quant_type'] == 'rv')]['quant1_err'].copy()

            self.epoch_rv_idx = [np.argmin(self.rv_observed),
                                 np.argmax(self.rv_observed)]

        # create an empty results object
        self.results = orbitize.results.Results(
            self.system,
            sampler_name=self.__class__.__name__,
            post=None,
            lnlike=None,
            version_number=orbitize.__version__
        )

    def prepare_samples(self, num_samples):
        """
        Prepare some orbits for rejection sampling. This draws random orbits
        from priors, and performs scale & rotate.

        Args:
            num_samples (int): number of orbits to draw and scale & rotate for
                OFTI to run rejection sampling on

        Return:
            np.array: array of prepared samples. The first dimension has size of
            num_samples. This should be passed into ``OFTI.reject()``
        """

        # generate sample orbits
        samples = np.empty([len(self.priors), num_samples])
        for i in range(len(self.priors)):
            if hasattr(self.priors[i], "draw_samples"):
                samples[i, :] = self.priors[i].draw_samples(num_samples)
            else: # param is fixed & has no prior
                samples[i, :] = self.priors[i] * np.ones(num_samples)

        # Make Converison to Standard Basis:
        samples = self.system.basis.to_standard_basis(samples)
        
        for body_num in np.arange(self.system.num_secondary_bodies) + 1:

            sma = samples[self.system.basis.standard_basis_idx['sma{}'.format(body_num)],:]
            ecc = samples[self.system.basis.standard_basis_idx['ecc{}'.format(body_num)],:]
            inc = samples[self.system.basis.standard_basis_idx['inc{}'.format(body_num)],:]
            argp = samples[self.system.basis.standard_basis_idx['aop{}'.format(body_num)],:]
            lan = samples[self.system.basis.standard_basis_idx['pan{}'.format(body_num)],:]
            tau = samples[self.system.basis.standard_basis_idx['tau{}'.format(body_num)],:]
            plx = samples[self.system.basis.standard_basis_idx['plx'],:]
            if self.system.fit_secondary_mass:
                m0 = samples[self.system.basis.standard_basis_idx['m0'],:]
                m1 = samples[self.system.basis.standard_basis_idx['m{}'.format(body_num)],:]
                mtot = m0 + m1
            else:
                mtot = samples[self.system.basis.standard_basis_idx['mtot'],:]
                m1 = None
            
            min_epoch = self.epoch_idx[body_num - 1]
            if min_epoch is None:
                # Don't need to rotate and scale if no astrometric measurments for this body. Brute force rejection sampling
                continue

            period_prescale = np.sqrt(
                4*np.pi**2*(sma*u.AU)**3/(consts.G*(mtot*u.Msun))
            )
            period_prescale = period_prescale.to(u.day).value
            meananno = self.epochs[min_epoch]/period_prescale - tau

            # compute sep/PA of generated orbits
            ra, dec, _ = orbitize.kepler.calc_orbit(
                self.epochs[min_epoch], sma, ecc, inc, argp, lan, tau, plx, mtot, 
                tau_ref_epoch=0, mass_for_Kamp=m1
            )
            sep, pa = orbitize.system.radec2seppa(ra, dec) # sep[mas], PA[deg]

            # generate Gaussian offsets from observational uncertainties
            sep_offset = np.random.normal(
                0, self.sep_err[min_epoch], size=num_samples
            )
            pa_offset =  np.random.normal(
                0, self.pa_err[min_epoch], size=num_samples
            )

            # calculate correction factors
            sma_corr = (sep_offset + self.sep_observed[min_epoch])/sep
            lan_corr = (pa_offset + self.pa_observed[min_epoch] - pa)

            # perform scale-and-rotate
            sma *= sma_corr # [AU]
            lan += np.radians(lan_corr) # [rad]
            lan = (lan + 2 * np.pi) % (2 * np.pi)

            if self.system.restrict_angle_ranges:
                argp[lan >= np.pi] += np.pi
                argp = argp % (2 * np.pi)
                lan[lan >= np.pi] -= np.pi

            period_new = np.sqrt(
                4*np.pi**2*(sma*u.AU)**3/(consts.G*(mtot*u.Msun))
            )
            period_new = period_new.to(u.day).value

            tau = (self.epochs[min_epoch]/period_new - meananno) % 1

            # updates samples with new values of sma, pan, tau
            samples[self.system.basis.standard_basis_idx['sma{}'.format(body_num)],:] = sma
            samples[self.system.basis.standard_basis_idx['aop{}'.format(body_num)],:] = argp
            samples[self.system.basis.standard_basis_idx['pan{}'.format(body_num)],:] = lan
            samples[self.system.basis.standard_basis_idx['tau{}'.format(body_num)],:] = tau

        return samples

    def reject(self, samples):
        """
        Runs rejection sampling on some prepared samples.

        Args:
            samples (np.array): array of prepared samples. The first dimension \
                has size ``num_samples``. This should be the output of \
                ``prepare_samples()``.

        Return:
            tuple:

                np.array: a subset of ``samples`` that are accepted based on the
                data.

                np.array: the log likelihood values of the accepted orbits.

        """

        lnp = self._logl(samples)

        # we just want the chi2 term for rejection, so compute the Gaussian normalization term and remove it
        errs = np.array([self.system.data_table['quant1_err'],
                         self.system.data_table['quant2_err']]).T

        if self.has_corr:
            corrs = self.system.data_table['quant12_corr']
        else:
            corrs = None
        lnp_scaled = lnp - orbitize.lnlike.chi2_norm_term(errs, corrs)

        # account for user-set priors on PAN that were destroyed by scale-and-rotate
        for body_num in np.arange(self.system.num_secondary_bodies) + 1:

            pan_idx = self.system.basis.standard_basis_idx['pan{}'.format(body_num)]

            pan_prior = self.system.sys_priors[pan_idx]
            if pan_prior is not orbitize.priors.UniformPrior:

                # apply PAN prior
                lnp_scaled += pan_prior.compute_lnprob(samples[pan_idx,:])

            # prior is uniform but with different bounds that OFTI expects
            elif (pan_prior.minval != 0) or (
                (pan_prior.maxval != np.pi) or (pan_prior.maxval != 2*np.pi)
            ):
                
                samples_outside_pan_prior = np.where(
                    (samples[pan_idx,:] < pan_prior.minval) or 
                    (samples[pan_idx,:] > pan_prior.maxval)
                )[0]

                lnp_scaled[samples_outside_pan_prior] = -np.inf

        # reject orbits with probability less than a uniform random number
        random_samples = np.log(np.random.random(len(lnp)))
        saved_orbit_idx = np.where(lnp_scaled > random_samples)[0]
        saved_orbits = np.array([samples[:, i] for i in saved_orbit_idx])
        lnlikes = np.array([lnp[i] for i in saved_orbit_idx])

        return saved_orbits, lnlikes

    def _sampler_process(self, output, total_orbits, num_samples=10000, Value=0, lock=None):
        """
        Runs OFTI until it finds the number of total accepted orbits desired.
        Meant to be called by run_sampler.

        Args:
            output (manager.Queue): manager.Queue object to store results
            total_orbits (int): total number of accepted orbits desired by user
            num_samples (int): number of orbits to prepare for OFTI to run
                rejection sampling on
            Value (mp.Value(int)): global counter for the orbits generated
            lock: mp.lock object to prevent issues caused by access to shared
                  memory by multiple processes
        Returns:
            tuple:

                output_orbits (np.array): array of accepted orbits,
                    size: total_orbits

                output_lnlikes (np.array): array of log probabilities,
                    size: total_orbits

        """

        np.random.seed()

        n_orbits_saved = 0
        output_orbits = np.empty((total_orbits, len(self.priors)))
        output_lnlikes = np.empty(total_orbits)

        # add orbits to `output_orbits` until `total_orbits` are saved
        while n_orbits_saved < total_orbits:

            samples = self.prepare_samples(num_samples)
            accepted_orbits, lnlikes = self.reject(samples)

            if len(accepted_orbits) == 0:
                pass
            else:
                n_accepted = len(accepted_orbits)
                maxindex2save = np.min([n_accepted, total_orbits - n_orbits_saved])
                output_orbits[n_orbits_saved: n_orbits_saved +
                              n_accepted] = accepted_orbits[0:maxindex2save]
                output_lnlikes[n_orbits_saved: n_orbits_saved+n_accepted] = lnlikes[0:maxindex2save]
                n_orbits_saved += maxindex2save

                # add to the value of the global variable
                with lock:
                    Value.value += maxindex2save

        output.put((np.array(output_orbits), output_lnlikes))
        return (np.array(output_orbits), output_lnlikes)

    def run_sampler(self, total_orbits, num_samples=10000, num_cores=None):
        """
        Runs OFTI in parallel on multiple cores until we get the number of total accepted orbits we want.

        Args:
            total_orbits (int): total number of accepted orbits desired by user
            num_samples (int): number of orbits to prepare for OFTI to run
                rejection sampling on. Defaults to 10000.
            num_cores (int): the number of cores to run OFTI on. Defaults to
                number of cores availabe.
        Return:
            np.array: array of accepted orbits. Size: total_orbits.

        Written by: Vighnesh Nagpal(2019)

        """

        if num_cores!=1:
            if num_cores==None:
                num_cores=mp.cpu_count()
            
            results=[]
            # orbits_saved is a global counter for the number of orbits generated 
            orbits_saved=mp.Value('i',0)
            
            manager = mp.Manager()            
            output = manager.Queue()

            # setup the processes
            lock = mp.Lock()
            nrun_per_core = int(np.ceil(float(total_orbits)/float(num_cores)))

            processes = [
                mp.Process(
                    target=self._sampler_process,
                    args=(output, nrun_per_core, num_samples,
                          orbits_saved, lock)
                ) for x in range(num_cores)
            ]

            # start the processes
            for p in processes:
                p.start()

            # print out the number of orbits generated every second
            while orbits_saved.value < total_orbits:
                print(str(orbits_saved.value)+'/'+str(total_orbits)+' orbits found', end='\r')
                time.sleep(0.1)

            print(str(total_orbits)+'/'+str(total_orbits)+' orbits found', end='\r')

            # join the processes
            for p in processes:
                p.join()
            # get the results of each process from the queue
            for p in processes:
                results.append(output.get())

            # filling up the output_orbits array
            output_orbits = np.zeros((total_orbits, len(self.priors)))
            output_lnlikes = np.empty(total_orbits)
            pos = 0

            for p in results:
                num_to_fill = np.min([len(p[0]), total_orbits - pos])
                output_orbits[pos:pos+num_to_fill] = p[0][0:num_to_fill]
                output_lnlikes[pos:pos+num_to_fill] = p[1][0:num_to_fill]
                pos += num_to_fill

            self.results.add_samples(
                np.array(output_orbits),
                output_lnlikes
            )
            return output_orbits

        else:
            # this block is executed if num_cores=1
            n_orbits_saved = 0
            output_orbits = np.empty((total_orbits, len(self.priors)))
            output_lnlikes = np.empty(total_orbits)

            # add orbits to `output_orbits` until `total_orbits` are saved
            while n_orbits_saved < total_orbits:
                samples = self.prepare_samples(num_samples)
                accepted_orbits, lnlikes = self.reject(samples)

                if len(accepted_orbits) == 0:
                    pass
                else:
                    n_accepted = len(accepted_orbits)
                    maxindex2save = np.min([n_accepted, total_orbits - n_orbits_saved])

                    output_orbits[n_orbits_saved: n_orbits_saved +
                                  n_accepted] = accepted_orbits[0:maxindex2save]
                    output_lnlikes[n_orbits_saved: n_orbits_saved +
                                   n_accepted] = lnlikes[0:maxindex2save]
                    n_orbits_saved += maxindex2save

                    # print progress statement
                    print(str(n_orbits_saved)+'/'+str(total_orbits)+' orbits found', end='\r')

            self.results.add_samples(
                np.array(output_orbits),
                output_lnlikes
            )

            return output_orbits


class MCMC(Sampler):
    """
    MCMC sampler. Supports either parallel tempering or just regular MCMC. Parallel tempering will be run if ``num_temps`` > 1
    Parallel-Tempered MCMC Sampler uses ptemcee, a fork of the emcee Affine-infariant sampler
    Affine-Invariant Ensemble MCMC Sampler uses emcee.

    .. Warning:: may not work well for multi-modal distributions

    Args:
        system (system.System): system.System object
        num_temps (int): number of temperatures to run the sampler at. 
            Parallel tempering will be used if num_temps > 1 (default=20)
        num_walkers (int): number of walkers at each temperature (default=1000)
        num_threads (int): number of threads to use for parallelization (default=1)
        chi2_type (str, optional): either  "standard", or "log"
        like (str): name of likelihood function in ``lnlike.py``
        custom_lnlike (func): ability to include an addition custom likelihood 
            function in the fit. The function looks like 
            ``clnlikes = custon_lnlike(params)`` where ``params`` is a RxM array 
            of fitting parameters, where R is the number of orbital paramters 
            (can be passed in system.compute_model()), and M is the number of 
            orbits we need model predictions for. It returns ``clnlikes`` 
            which is an array of length M, or it can be a single float if M = 1.
        prev_result_filename (str): if passed a filename to an HDF5 file 
            containing a orbitize.Result data, MCMC will restart from where it 
            left off. 

    Written: Jason Wang, Henry Ngo, 2018
    """
    def __init__(
        self, system, num_temps=20, num_walkers=1000, num_threads=1, chi2_type='standard', 
        like='chi2_lnlike', custom_lnlike=None, prev_result_filename=None
    ):

        super(MCMC, self).__init__(system, like=like, chi2_type=chi2_type, custom_lnlike=custom_lnlike)

        self.num_temps = num_temps
        self.num_walkers = num_walkers
        self.num_threads = num_threads

        # create an empty results object
        self.results = orbitize.results.Results(
            self.system,
            sampler_name=self.__class__.__name__,
            post=None,
            lnlike=None,
            version_number=orbitize.__version__
        )

        if self.num_temps > 1:
            self.use_pt = True
        else:
            self.use_pt = False
            self.num_temps = 1

        # get priors from the system class. need to remove and record fixed priors
        self.priors = []
        self.fixed_params = []

        self.sampled_param_idx = {}
        sampled_param_counter = 0
        for i, prior in enumerate(system.sys_priors):

            # check for fixed parameters
            if not hasattr(prior, "draw_samples"):
                self.fixed_params.append((i, prior))
            else:
                self.priors.append(prior)
                self.sampled_param_idx[self.system.labels[i]] = sampled_param_counter
                sampled_param_counter += 1

        # initialize walkers initial postions
        self.num_params = len(self.priors)

        if prev_result_filename is None:
            # initialize walkers initial postions

            init_positions = []
            for prior in self.priors:
                # draw them uniformly becase we don't know any better right now
                # TODO: be smarter in the future
                random_init = prior.draw_samples(num_walkers*self.num_temps)
                if self.num_temps > 1:
                    random_init = random_init.reshape([self.num_temps, num_walkers])

                init_positions.append(random_init)

            # save this as the current position for the walkers
            if self.use_pt:
                # make this an numpy array, but combine the parameters into a shape of (ntemps, nwalkers, nparams)
                # we currently have a list of [ntemps, nwalkers] with nparam arrays. We need to make nparams the third dimension
                self.curr_pos = np.dstack(init_positions)
            else:
                # make this an numpy array, but combine the parameters into a shape of (nwalkers, nparams)
                # we currently have a list of arrays where each entry is num_walkers prior draws for each parameter
                # We need to make nparams the second dimension, so we have to transpose the stacked array
                self.curr_pos = np.stack(init_positions).T
        else:
            # restart from previous walker positions
            self.results.load_results(prev_result_filename, append=True)

            prev_pos = self.results.curr_pos

            # check previous positions has the correct dimensions as we need given how this sampler was created.
            expected_shape = (self.num_walkers, len(self.priors))
            if self.use_pt:
                expected_shape = (self.num_temps,) + expected_shape
            if prev_pos.shape != expected_shape:
                raise ValueError("Unable to restart chain. Saved walker positions has shape {0}, while current sampler needs {1}".format(prev_pos.shape, expected_shape))

            self.curr_pos = prev_pos

    def _fill_in_fixed_params(self, sampled_params):
        """
        Fills in the missing parameters from the chain that aren't being sampled

        Args:
            sampled_params (np.array): either 1-D array of size = number of 
                sampled params, or 2-D array of shape (num_models, num_params)

        Returns:
            np.array: same number of dimensions as sampled_params, 
                but with num_params including the fixed parameters
        """
        if len(self.fixed_params) == 0:
            # nothing to add
            return sampled_params

        # check if 1-D or 2-D
        twodim = np.ndim(sampled_params) == 2

        # insert in params
        for index, value in self.fixed_params:
            if twodim:
                sampled_params = np.insert(sampled_params, index, value, axis=1)
            else:
                sampled_params = np.insert(sampled_params, index, value)

        return sampled_params

    def _logl(self, params, include_logp=False):
        """
        log likelihood function that interfaces with the orbitize objects
        Comptues the sum of the log likelihoods of the data given the input model

        Args:
            params (np.array of float): MxR array
                of fitting parameters, where R is the number of
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in System() above. If M=1, this can be a 1d array.
            include_logp (bool): if True, also include log prior in this function

        Returns:
            lnlikes (float): sum of all log likelihoods of the data given input model

        """
        if include_logp:
            if np.ndim(params) == 1:
                logp = orbitize.priors.all_lnpriors(params, self.priors)
                # escape if logp == -np.inf
                if np.isinf(logp):
                    return -np.inf
            else:
                logp = np.array([orbitize.priors.all_lnpriors(pset, self.priors)
                                 for pset in params])
        else:
            logp = 0  # don't include prior

        full_params = self._fill_in_fixed_params(params)
        if np.ndim(full_params) == 2:
            full_params = full_params.T

        return super(MCMC, self)._logl(full_params) + logp

    def _update_chains_from_sampler(self, sampler, num_steps=None):
        """
        Updates self.post, self.chain, and self.lnlike from the MCMC sampler

        Args:
            sampler (emcee.EnsembleSampler or ptemcee.Sampler): sampler object.
            num_steps (int): if not None, only stores the first num_steps number of steps
        """
        if num_steps is None:
            # use all the steps, grab total number of steps from dimension of chains
            num_steps = sampler.chain.shape[-2]

        self.chain = sampler.chain
        num_params = self.chain.shape[-1]

        if self.use_pt:
            # chain is shape: Ntemp x Nwalkers x Nsteps x Nparams
            self.post = sampler.chain[0, :, :num_steps].reshape(-1, num_params) # the reshaping flattens the chain
            # should also be picking out the lowest temperature logps
            self.lnlikes = sampler.loglikelihood[0, :, :num_steps].flatten()
            self.lnlikes_alltemps = sampler.loglikelihood[:, :, :num_steps]
        else:
            # chain is shape: Nwalkers x Nsteps x Nparams
            self.post = sampler.chain[:, :num_steps].reshape(-1, num_params)
            self.lnlikes = sampler.lnprobability[:, :num_steps].flatten()

            # convert posterior probability (returned by sampler objects) to likelihood (required by orbitize.results.Results)
            for i, orb in enumerate(self.post):
                self.lnlikes[i] -= orbitize.priors.all_lnpriors(orb, self.priors)

        # include fixed parameters in posterior
        self.post = self._fill_in_fixed_params(self.post)

    def validate_xyz_positions(self):
        """
        If using the XYZ basis, walkers might be initialized in an invalid 
        region of parameter space. This function fixes that by replacing invalid 
        positions by new randomly generated positions until all are valid.
        """
        if self.system.fitting_basis == 'XYZ':
            if self.use_pt:
                all_valid = False
                while not all_valid:
                    total_invalids = 0
                    for temp in range(self.num_temps):
                        to_stand = self.system.basis.to_standard_basis(self.curr_pos[temp,:,:].T.copy()).T

                        # Get invalids by checking ecc values for each companion
                        indices = [((i * 6) + 1) for i in range(self.system.num_secondary_bodies)]
                        invalids = np.where((to_stand[:, indices] < 0.) | (to_stand[:, indices] >= 1.))[0]

                        # Redraw samples for the invalid ones
                        if len(invalids) > 0:
                            newpos = []
                            for prior in self.priors:
                                randompos = prior.draw_samples(len(invalids))
                                newpos.append(randompos)
                            self.curr_pos[temp, invalids, :] = np.stack(newpos).T 
                            total_invalids += len(invalids)
                    if total_invalids == 0:
                        all_valid = True
                        print('All walker positions validated.')
            else:
                all_valid = False
                while not all_valid:
                    total_invalids = 0
                    to_stand = self.system.basis.to_standard_basis(self.curr_pos[:,:].T.copy()).T

                    # Get invalids by checking ecc values for each companion
                    indices = [((i * 6) + 1) for i in range(self.system.num_secondary_bodies)]
                    invalids = np.where((to_stand[:, indices] < 0.) | (to_stand[:, indices] >= 1.))[0]                    

                    # Redraw saples for the invalid ones
                    if len(invalids) > 0:
                        newpos = []
                        for prior in self.priors:
                            randompos = prior.draw_samples(len(invalids))
                            newpos.append(randompos)
                        self.curr_pos[invalids, :] = np.stack(newpos).T 
                        total_invalids += len(invalids)
                    if total_invalids == 0:
                        all_valid = True
                        print('All walker positions validated.')



    def run_sampler(
        self, total_orbits, burn_steps=0, thin=1, examine_chains=False, 
        output_filename=None, periodic_save_freq=None
    ):
        """
        Runs PT MCMC sampler. Results are stored in ``self.chain`` and ``self.lnlikes``.
        Results also added to ``orbitize.results.Results`` object (``self.results``)

        .. Note:: Can be run multiple times if you want to pause and inspect things.
            Each call will continue from the end state of the last execution.

        Args:
            total_orbits (int): total number of accepted possible
                orbits that are desired. This equals
                ``num_steps_per_walker`` x ``num_walkers``
            burn_steps (int): optional paramter to tell sampler
                to discard certain number of steps at the beginning
            thin (int): factor to thin the steps of each walker
                by to remove correlations in the walker steps
            examine_chains (boolean): Displays plots of walkers at each step by
                running `examine_chains` after `total_orbits` sampled.
            output_filename (str): Optional filepath for where results file can be saved.
            periodic_save_freq (int): Optionally, save the current results into ``output_filename``
                every nth step while running, where n is value passed into this variable.

        Returns:
            ``emcee.sampler`` object: the sampler used to run the MCMC
        """

        if periodic_save_freq is not None and output_filename is None:
            raise ValueError("output_filename must be defined for periodic saving of the chains")
        if periodic_save_freq is not None and not isinstance(periodic_save_freq, int):
            raise TypeError("periodic_save_freq must be an integer")

        nsteps = int(np.ceil(total_orbits / self.num_walkers))
        if nsteps <= 0:
            raise ValueError("Total_orbits must be greater than num_walkers.")

        with Pool(processes=self.num_threads) as pool: 
            if self.use_pt:
                sampler = ptemcee.Sampler(
                    self.num_walkers, self.num_params, self._logl, orbitize.priors.all_lnpriors,
                    ntemps=self.num_temps, threads=self.num_threads, logpargs=[self.priors, ]
                )
            else:
                sampler = emcee.EnsembleSampler(
                    self.num_walkers, self.num_params, self._logl, pool=pool,
                    kwargs={'include_logp': True}
                )

            print("Starting Burn in")
            for i, state in enumerate(sampler.sample(self.curr_pos, iterations=burn_steps, thin=thin)):
                if self.use_pt:
                    self.curr_pos = state[0]
                else:
                    self.curr_pos = state.coords

                if (i+1) % 5 == 0:
                    print(str(i+1)+'/'+str(burn_steps)+' steps of burn-in complete', end='\r')

                if periodic_save_freq is not None:
                    if (i+1) % periodic_save_freq == 0: # we've completed i+1 steps
                        self.results.curr_pos = self.curr_pos
                        self.results.save_results(output_filename)

            sampler.reset()
            print('')
            print('Burn in complete. Sampling posterior now.')

            saved_upto = 0 # keep track of how many steps of this chain we've saved. this is the next index that needs to be saved 
            for i, state in enumerate(sampler.sample(self.curr_pos, iterations=nsteps, thin=thin)):
                if self.use_pt:
                    self.curr_pos = state[0]
                else:
                    self.curr_pos = state.coords
                    
                # print progress statement
                if (i+1) % 5 == 0:
                    print(str(i+1)+'/'+str(nsteps)+' steps completed', end='\r')

                if periodic_save_freq is not None:
                    if (i+1) % periodic_save_freq == 0: # we've completed i+1 steps
                        self._update_chains_from_sampler(sampler, num_steps=i+1)

                        # figure out what is the new chunk of the chain and corresponding lnlikes that have been computed before last save
                        # grab the current posterior and lnlikes and reshape them to have the Nwalkers x Nsteps dimension again
                        post_shape = self.post.shape
                        curr_chain_shape = (self.num_walkers, post_shape[0]//self.num_walkers, post_shape[-1])
                        curr_chain = self.post.reshape(curr_chain_shape)
                        curr_lnlike_chain = self.lnlikes.reshape(curr_chain_shape[:2])
                        # use the reshaped arrays and find the new steps we computed
                        curr_chunk = curr_chain[:, saved_upto:i+1]
                        curr_chunk = curr_chunk.reshape(-1, curr_chunk.shape[-1]) # flatten nwalkers x nsteps dim
                        curr_lnlike_chunk = curr_lnlike_chain[:, saved_upto:i+1].flatten()

                        # add this current chunk to the results object (which already has all the previous chunks saved)
                        self.results.add_samples(curr_chunk, curr_lnlike_chunk, 
                                                    curr_pos=self.curr_pos)
                        self.results.save_results(output_filename)
                        saved_upto = i+1

            print('')
            self._update_chains_from_sampler(sampler)

            if periodic_save_freq is None:
                # need to save everything
                self.results.add_samples(self.post, self.lnlikes, curr_pos=self.curr_pos)
            elif saved_upto < nsteps:
                # just need to save the last few
                # same code as above except we just need to grab the last few
                post_shape = self.post.shape
                curr_chain_shape = (self.num_walkers, post_shape[0]//self.num_walkers, post_shape[-1])
                curr_chain = self.post.reshape(curr_chain_shape)
                curr_lnlike_chain = self.lnlikes.reshape(curr_chain_shape[:2])
                curr_chunk = curr_chain[:, saved_upto:]
                curr_chunk = curr_chunk.reshape(-1, curr_chunk.shape[-1]) # flatten nwalkers x nsteps dim
                curr_lnlike_chunk = curr_lnlike_chain[:, saved_upto:].flatten()

                self.results.add_samples(curr_chunk, curr_lnlike_chunk, 
                                                     curr_pos=self.curr_pos)

            if output_filename is not None:
                self.results.save_results(output_filename)

            print('Run complete')
        # Close pool
        if examine_chains:
            self.examine_chains()

        return sampler

    def examine_chains(self, param_list=None, walker_list=None, n_walkers=None, step_range=None, transparency = 1):
        """
        Plots position of walkers at each step from Results object. Returns list of figures, one per parameter
        Args:
            param_list: List of strings of parameters to plot (e.g. "sma1")
                If None (default), all parameters are plotted
            walker_list: List or array of walker numbers to plot
                If None (default), all walkers are plotted
            n_walkers (int): Randomly select `n_walkers` to plot
                Overrides walker_list if this is set
                If None (default), walkers selected as per `walker_list`
            step_range (array or tuple): Start and end values of step numbers to plot
                If None (default), all the steps are plotted
            transparency (int or float): Determines visibility of the plotted function
                If 1 (default) results plot at 100% opacity

        Returns:
            List of ``matplotlib.pyplot.Figure`` objects:
                Walker position plot for each parameter selected

        (written): Henry Ngo, 2019
        """

        # Get the flattened chain from Results object (nwalkers*nsteps, nparams)
        flatchain = np.copy(self.results.post)
        total_samples, n_params = flatchain.shape
        n_steps = np.int(total_samples/self.num_walkers)
        # Reshape it to (nwalkers, nsteps, nparams)
        chn = flatchain.reshape((self.num_walkers, n_steps, n_params))

        # Get list of walkers to use
        if n_walkers is not None:  # If n_walkers defined, randomly choose that many walkers
            walkers_to_plot = np.random.choice(self.num_walkers, size=n_walkers, replace=False)
        elif walker_list is not None:  # if walker_list is given, use that list
            walkers_to_plot = np.array(walker_list)
        else:  # both n_walkers and walker_list are none, so use all walkers
            walkers_to_plot = np.arange(self.num_walkers)

        # Get list of parameters to use
        if param_list is None:
            params_to_plot = np.arange(n_params)
        else:  # build list from user input strings
            params_plot_list = []
            for i in param_list:
                if i in self.system.basis.param_idx:
                    params_plot_list.append(self.system.basis.param_idx[i])
                else:
                    raise Exception('Invalid param name: {}. See system.basis.param_idx.'.format(i))
            params_to_plot = np.array(params_plot_list)

        # Loop through each parameter and make plot
        output_figs = []
        for pp in params_to_plot:
            fig, ax = plt.subplots()
            for ww in walkers_to_plot:
                ax.plot(chn[ww, :, pp], 'k-', alpha = transparency)
            ax.set_xlabel('Step')
            if step_range is not None:  # Limit range shown if step_range is set
                ax.set_xlim(step_range)
            output_figs.append(fig)

        # Return
        return output_figs

    def chop_chains(self, burn, trim=0):
        """
        Permanently removes steps from beginning (and/or end) of chains from the 
        Results object. Also updates `curr_pos` if steps are removed from the 
        end of the chain.

        Args:
            burn (int): The number of steps to remove from the beginning of the chains
            trim (int): The number of steps to remove from the end of the chians (optional)

        .. Warning:: Does not update bookkeeping arrays within `MCMC` sampler object.

        (written): Henry Ngo, 2019
        """

        # Retrieve information from results object
        flatchain = np.copy(self.results.post)
        total_samples, n_params = flatchain.shape
        n_steps = np.int(total_samples/self.num_walkers)
        flatlnlikes = np.copy(self.results.lnlike)

        # Reshape chain to (nwalkers, nsteps, nparams)
        chn = flatchain.reshape((self.num_walkers, n_steps, n_params))
        # Reshape lnlike to (nwalkers, nsteps)
        lnlikes = flatlnlikes.reshape((self.num_walkers, n_steps))

        # Find beginning and end indices for steps to keep
        keep_start = burn
        keep_end = n_steps - trim
        n_chopped_steps = n_steps - trim - burn

        # Update arrays in `sampler`: chain, lnlikes, lnlikes_alltemps (if PT), post
        chopped_chain = chn[:, keep_start:keep_end, :]
        chopped_lnlikes = lnlikes[:, keep_start:keep_end]

        # Update current position if trimmed from edge
        if trim > 0:
            self.curr_pos = chopped_chain[:, -1, :]

        # Flatten likelihoods and samples
        flat_chopped_chain = chopped_chain.reshape(self.num_walkers*n_chopped_steps, n_params)
        flat_chopped_lnlikes = chopped_lnlikes.reshape(self.num_walkers*n_chopped_steps)

        # Update results object associated with this sampler
        self.results = orbitize.results.Results(
            self.system, 
            sampler_name=self.__class__.__name__,
            post=flat_chopped_chain,
            lnlike=flat_chopped_lnlikes,
            version_number = orbitize.__version__,
            curr_pos = self.curr_pos
        )

        # Print a confirmation
        print('Chains successfully chopped. Results object updated.')

    def check_prior_support(self):
        """
        Review the positions of all MCMC walkers, to verify that they are supported by the prior space.
        This function will raise a descriptive ValueError if any positions lie outside prior support.
        Otherwise, it will return nothing.

        (written): Adam Smith, 2021
        """

        # Flatten the walker/temperature positions for ease of manipulation.
        all_positions = self.curr_pos.reshape(self.num_walkers*self.num_temps,self.num_params)
        
        # Placeholder list to track any bad parameters that come up.
        bad_parameters = []

        # If there are no covarient priors, loop on each variable to locate any out-of-place parameters. (this is why we transpose the walkers)
        if not np.any([prior.is_correlated for prior in self.priors]):
            for i, x in enumerate(all_positions.T):
                # Any issues with this parameter?
                lnprob = self.priors[i].compute_lnprob(np.array(x))
                supported = np.isfinite(lnprob).all() == True

                if supported == False:
                    # Problem detected. Take note and continue the loop - we want to catch all the problem parameters.
                    bad_parameters.append(str(i))

            # Throw our ValueError if necessary,
            if len(bad_parameters) > 0:
                raise ValueError("Attempting to start with walkers outside of prior support: check parameter(s) "+', '.join(bad_parameters))

        # We're not done yet, however. There may be errors in covariant priors; run a check for that.
        else:
            for y in all_positions:
                lnprob = orbitize.priors.all_lnpriors(y,self.priors)
                if not np.isfinite(lnprob).all():
                    raise ValueError("Attempting to start with walkers outside of prior support: covariant prior failure.")
        
        # otherwise exit the function and continue.
        return
