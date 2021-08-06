import numpy as np
import astropy.units as u, astropy.constants as consts
import warnings # remove when functions are depreciated
import abc
import pdb

from orbitize import priors
from scipy.optimize import fsolve

class Basis(abc.ABC):
    '''
    Abstract base class for different basis sets. All new basis objects should inherit from
    this class. This class is meant to control how priors are assigned to various basis sets
    and how conversions are made from the basis sets to the standard keplarian set.
    '''

    def __init__(self, stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
        angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, rv_instruments=None):

        self.stellar_mass = stellar_mass
        self.mass_err=mass_err
        self.plx=plx
        self.plx_err=plx_err
        self.num_secondary_bodies=num_secondary_bodies
        self.angle_upperlim=angle_upperlim
        self.fit_secondary_mass=fit_secondary_mass
        self.hipparcos_IAD = hipparcos_IAD
        self.rv = rv
        self.rv_instruments = rv_instruments

        # Define dictionary of default priors to be updated as new basis sets are added
        self.default_priors = {'sma' : priors.LogUniformPrior(0.001, 1e4), 'per' : priors.LogUniformPrior(1e-5, 1e6),
                               'ecc' : priors.UniformPrior(0., 1.), 'inc' : priors.SinPrior(), 'aop' : priors.UniformPrior(0., 2.*np.pi),
                               'pan' : priors.UniformPrior(0., angle_upperlim), 'tau' : priors.UniformPrior(0., 1.),
                               'K' : priors.LogUniformPrior(1e-4, 10)}
        

    @abc.abstractmethod
    def construct_priors(self):
        pass

    @abc.abstractmethod
    def to_standard_basis(self, param_arr):
        pass

    def set_hip_iad_priors(self, priors_arr, labels_arr):
        '''
        Adds the necessary priors relevant to the hipparcos data to 'priors_arr' and updates 'labels_arr'
        with the priors' corresponding labels.

        Args:
            priors_arr (list of orbitize.priors.Prior objects): holds the prior objects for each parameter to be fitted (updated here)
            labels_arr (list of strings): holds the name of all the parameters to be fitted (updated here)
        '''

        priors_arr.append(priors.UniformPrior(
            self.hipparcos_IAD.pm_ra0 - 10 * self.hipparcos_IAD.pm_ra0_err,
            self.hipparcos_IAD.pm_ra0 + 10 * self.hipparcos_IAD.pm_ra0_err)
        )
        labels_arr.append('pm_ra')

        priors_arr.append(priors.UniformPrior(
            self.hipparcos_IAD.pm_dec0 - 10 * self.hipparcos_IAD.pm_dec0_err,
            self.hipparcos_IAD.pm_dec0 + 10 * self.hipparcos_IAD.pm_dec0_err)
        )
        labels_arr.append('pm_dec')

        priors_arr.append(priors.UniformPrior(
            - 10 * self.hipparcos_IAD.alpha0_err,
            10 * self.hipparcos_IAD.alpha0_err)
        )
        labels_arr.append('alpha0')

        priors_arr.append(priors.UniformPrior(
            - 10 * self.hipparcos_IAD.delta0_err,
            10 * self.hipparcos_IAD.delta0_err)
        )
        labels_arr.append('delta0')

    def set_rv_priors(self, priors_arr, labels_arr):
        '''
        Adds the necessary priors if radial velocity data is supplied to 'priors_arr' and updates 'labels_arr'
        with the priors' corresponding labels. This function assumes that 'rv' data has been supplied and 
        a secondary mass is being fitted for.

        Args:
            priors_arr (list of orbitize.priors.Prior objects): holds the prior objects for each parameter to be fitted (updated here)
            labels_arr (list of strings): holds the name of all the parameters to be fitted (updated here)
        '''        

        for instrument in self.rv_instruments:
            priors_arr.append(priors.UniformPrior(-5, 5))  # gamma prior in km/s
            labels_arr.append('gamma_{}'.format(instrument))

            priors_arr.append(priors.LogUniformPrior(1e-4, 0.05))  # jitter prior in km/s
            labels_arr.append('sigma_{}'.format(instrument))

    def set_default_mass_priors(self, priors_arr, labels_arr):
        '''
        Adds the necessary priors for the stellar and/or companion masses.

        Args:
            priors_arr (list of orbitize.priors.Prior objects): holds the prior objects for each parameter to be fitted (updated here)
            labels_arr (list of strings): holds the name of all the parameters to be fitted (updated here)
        '''
        if self.fit_secondary_mass:
            for body in np.arange(self.num_secondary_bodies)+1:
                priors_arr.append(priors.LogUniformPrior(1e-6, 2)) # in Solar masses
                labels_arr.append('m{}'.format(body))
            labels_arr.append('m0')
        else:
            labels_arr.append('mtot')

        if self.mass_err > 0:
            priors_arr.append(priors.GaussianPrior(self.stellar_mass, self.mass_err))
        else:
            priors_arr.append(self.stellar_mass)


class Standard(Basis):
    '''
    Standard basis set based upon the 6 standard Keplarian elements: (sma, ecc, inc, aop, pan, tau).

    NOTE: 
        This class has support with both OFTI and MCMC.

    Args:
        stellar_mass (float): mean mass of the primary, in M_sol
        mass_err (float): uncertainty on 'stellar_mass', in M_sol
        plx (float): mean parallax of the system, in mas
        plx_err (float): uncertainty on 'plx', in mas
        num_secondary_bodies (int): number of secondary bodies in the system, should be at least 1
        fit_secondary_mass (bool): if True, include the dynamical mass of orbitting body as fitted parameter, if False,
            'stellar_mass' is taken to be total mass
        angle_upperlim (float): either pi or 2pi, to restrict the prior range for 'pan' parameter (default: 2*pi)
        hipparcos_IAD (orbitize.HipparcosLogProb object): if not 'None', then add relevant priors to this data (default: None)
        rv (bool): if True, then there is radial velocity data and assign radial velocity priors, if False, then there
            is no radial velocity data and radial velocity priors are not assigned (default: False)
        rv_instruments (np.array): array of unique rv instruments from the originally supplied data (default: None)
    '''

    def __init__(self, stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
        angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, rv_instruments=None):

        super(Standard, self).__init__(stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, 
            fit_secondary_mass, angle_upperlim, hipparcos_IAD, rv, rv_instruments)

    def construct_priors(self):
        base_labels = ['sma', 'ecc', 'inc', 'aop', 'pan', 'tau']
        basis_priors = []
        basis_labels = []

        # Add the priors common to each companion
        for body in np.arange(self.num_secondary_bodies):
            for elem in base_labels:
                basis_priors.append(self.default_priors[elem])
                basis_labels.append(elem + str(body+1))

        # Add parallax prior
        basis_labels.append('plx')
        if self.plx_err > 0:
            basis_priors.append(priors.GaussianPrior(self.plx, self.plx_err))
        else:
            basis_priors.append(self.plx)

        # Add hippparcos priors if necessary
        if self.hipparcos_IAD is not None:
            self.set_hip_iad_priors()

        # Add rv priors
        if self.rv and self.fit_secondary_mass:
            self.set_rv_priors(basis_priors, basis_labels)

        # Add mass priors
        self.set_default_mass_priors(basis_priors, basis_labels)

        return basis_priors, basis_labels

    def to_standard_basis(self, param_arr):
        return param_arr

class Period(Basis):
    '''
    Modification of the standard basis, swapping our sma for period: (per, ecc, inc, aop, pan, tau).

    NOTE: 
        This class does not have support for OFTI yet.

    Args:
        stellar_mass (float): mean mass of the primary, in M_sol
        mass_err (float): uncertainty on 'stellar_mass', in M_sol
        plx (float): mean parallax of the system, in mas
        plx_err (float): uncertainty on 'plx', in mas
        num_secondary_bodies (int): number of secondary bodies in the system, should be at least 1
        fit_secondary_mass (bool): if True, include the dynamical mass of orbitting body as fitted parameter, if False,
            'stellar_mass' is taken to be total mass
        angle_upperlim (float): either pi or 2pi, to restrict the prior range for 'pan' parameter (default: 2*pi)
        hipparcos_IAD (orbitize.HipparcosLogProb object): if not 'None', then add relevant priors to this data (default: None)
        rv (bool): if True, then there is radial velocity data and assign radial velocity priors, if False, then there
            is no radial velocity data and radial velocity priors are not assigned (default: False)
        rv_instruments (np.array): array of unique rv instruments from the originally supplied data (default: None)
    '''

    def __init__(self, stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
        angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, rv_instruments=None):

        super(Period, self).__init__(stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, 
            fit_secondary_mass, angle_upperlim, hipparcos_IAD, rv, rv_instruments)

    def construct_priors(self):
        base_labels = ['per', 'ecc', 'inc', 'aop', 'pan', 'tau']
        basis_priors = []
        basis_labels = []

        # Add the priors common to each companion
        for body in np.arange(self.num_secondary_bodies):
            for elem in base_labels:
                basis_priors.append(self.default_priors[elem])
                basis_labels.append(elem + str(body+1))

        # Add parallax prior
        basis_labels.append('plx')
        if self.plx_err > 0:
            basis_priors.append(priors.GaussianPrior(self.plx, self.plx_err))
        else:
            basis_priors.append(self.plx)

        # Add hipparcos priors if necessary
        if self.hipparcos_IAD is not None:
            self.set_hip_iad_priors()

        # Add rv priors
        if self.rv and self.fit_secondary_mass:
            self.set_rv_priors(basis_priors, basis_labels)

        # Add mass priors
        self.set_default_mass_priors(basis_priors, basis_labels)

        return basis_priors, basis_labels

    def to_standard_basis(self, param_arr):
        for i in np.arange(self.num_secondary_bodies)+1:
            startindex = 6 * (i - 1)
            per = param_arr[startindex]
            mtot = param_arr[-1]

            if self.fit_secondary_mass:
                # Assume two-body system
                m_secondary = param_arr[-1-self.num_secondary_bodies+(i-1)]
                m0 = param_arr[-1]
                mtot = m_secondary + m0

            sma = np.cbrt((consts.G*(mtot*u.Msun)*((per*u.year)**2))/(4*np.pi**2))
            sma = sma.to(u.AU).value
            param_arr[startindex] = sma

        return param_arr

class SemiAmp(Basis):
    '''
    Modification of the standard basis, swapping our sma for period and additionally sampling in
    the stellar radial velocity semi-amplitude: (per, ecc, inc, aop, pan, tau, K).

    NOTES: 
        This class does not have support for OFTI yet.
        Ideally, 'fit_secondary_mass' is true and rv data is supplied.

    Args:
        stellar_mass (float): mean mass of the primary, in M_sol
        mass_err (float): uncertainty on 'stellar_mass', in M_sol
        plx (float): mean parallax of the system, in mas
        plx_err (float): uncertainty on 'plx', in mas
        num_secondary_bodies (int): number of secondary bodies in the system, should be at least 1
        fit_secondary_mass (bool): if True, include the dynamical mass of orbitting body as fitted parameter, if False,
            'stellar_mass' is taken to be total mass
        angle_upperlim (float): either pi or 2pi, to restrict the prior range for 'pan' parameter (default: 2*pi)
        hipparcos_IAD (orbitize.HipparcosLogProb object): if not 'None', then add relevant priors to this data (default: None)
        rv (bool): if True, then there is radial velocity data and assign radial velocity priors, if False, then there
            is no radial velocity data and radial velocity priors are not assigned (default: False)
        rv_instruments (np.array): array of unique rv instruments from the originally supplied data (default: None)
    '''

    def __init__(self, stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
        angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, rv_instruments=None):

        super(SemiAmp, self).__init__(stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, 
            fit_secondary_mass, angle_upperlim, hipparcos_IAD, rv, rv_instruments)

    def construct_priors(self):
        base_labels = ['per', 'ecc', 'inc', 'aop', 'pan', 'tau', 'K']
        basis_priors = []
        basis_labels = []

        # Add the priors common to each companion
        for body in np.arange(self.num_secondary_bodies):
            for elem in base_labels:
                basis_priors.append(self.default_priors[elem])
                basis_labels.append(elem + str(body+1))


        # Add parallax prior
        basis_labels.append('plx')
        if self.plx_err > 0:
            basis_priors.append(priors.GaussianPrior(self.plx, self.plx_err))
        else:
            basis_priors.append(self.plx)

        # Add hip_iad priors if necessary
        if self.hipparcos_IAD is not None:
            self.set_hip_iad_priors()

        # Add rv priors
        if self.rv:
            self.set_rv_priors(basis_priors, basis_labels)

        # Add star mass prior (for now, regardless of whether 'fit_secondary_mass' is true)
        if self.mass_err > 0:
            basis_priors.append(priors.GaussianPrior(self.stellar_mass, self.mass_err))
        else:
            basis_priors.append(self.stellar_mass)

        basis_labels.append('m0')

        return basis_priors, basis_labels

    def to_standard_basis(self, param_arr):
        indices_to_remove = []  # Keep track of where semi-amp values are for removal
        m0 = param_arr[-1]
        base_labels_len = 7

        # Compute each companion's mass and sma
        for body in np.arange(self.num_secondary_bodies):
            period = param_arr[body * base_labels_len]
            ecc = param_arr[(body * base_labels_len) + 1]
            inc = param_arr[(body * base_labels_len) + 2]
            semi_amp = param_arr[(body * base_labels_len) + 6]
            indices_to_remove.append((body * base_labels_len) + 6)

            # Add companion mass and replace period with sma
            companion_m = self.compute_companion_mass(period, ecc, inc, semi_amp, m0)
            param_arr = np.insert(param_arr, -1, companion_m)
            companion_sma = self.compute_companion_sma(period, m0, companion_m)
            param_arr[body * base_labels_len] = companion_sma

        # Remove semi-amplitude values
        param_arr = np.delete(param_arr, indices_to_remove)

        return param_arr

    def func(self, x, lhs, m0):
        return ((x / ((x + m0)**(2/3))) - lhs)

    def compute_companion_mass(self, period, ecc, inc, semi_amp, m0):
        # Define LHS of equation
        kms = u.km / u.s
        lhs = ((semi_amp*kms)*((1-ecc**2)**(1/2))*((period*u.yr)**(1/3))*(consts.G**(-1/3))*((4*np.pi**2)**(-1/6))) / (np.sin(inc))
        lhs = (lhs.to((u.solMass)**(1/3))).value

        # Make guess at the middle of default prior space for companion mass (possibly iterative approach)
        m_n = fsolve(self.func, x0=1e-3, args=(lhs, m0))

        return m_n

    def compute_companion_sma(self, period, m0, m_n):
        sma = np.cbrt((consts.G*((m0+m_n)*u.Msun)*((period*u.yr)**2))/(4*np.pi**2))
        sma = sma.to(u.AU).value

        return sma

class XYZ(Basis):
    '''
    Defines an orbit using the companion's position and velocity components in XYZ space (x, y, z, xdot, ydot, zdot). 

    NOTES: 
        This class does not have support for OFTI yet.

    Args:
        stellar_mass (float): mean mass of the primary, in M_sol
        mass_err (float): uncertainty on 'stellar_mass', in M_sol
        plx (float): mean parallax of the system, in mas
        plx_err (float): uncertainty on 'plx', in mas
        num_secondary_bodies (int): number of secondary bodies in the system, should be at least 1
        fit_secondary_mass (bool): if True, include the dynamical mass of orbitting body as fitted parameter, if False,
            'stellar_mass' is taken to be total mass
        input_table (astropy.table.Table): output from 'orbitize.read_input.read_file()'
        best_epoch_idx (list): indices of the epochs corresponding to the smallest uncertainties
        epochs (list): all of the epochs from 'input_table'
        angle_upperlim (float): either pi or 2pi, to restrict the prior range for 'pan' parameter (default: 2*pi)
        hipparcos_IAD (orbitize.HipparcosLogProb object): if not 'None', then add relevant priors to this data (default: None)
        rv (bool): if True, then there is radial velocity data and assign radial velocity priors, if False, then there
            is no radial velocity data and radial velocity priors are not assigned (default: False)
        rv_instruments (np.array): array of unique rv instruments from the originally supplied data (default: None)

    Author: Rodrigo
    '''
    def __init__(self, stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
        input_table, best_epoch_idx, epochs, angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, rv_instruments=None):

        super(XYZ, self).__init__(stellar_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
            angle_upperlim, hipparcos_IAD, rv, rv_instruments)

        self.input_table = input_table
        self.best_epoch_idx = best_epoch_idx
        self.epochs = epochs

    def construct_priors(self):
        basis_priors = []
        basis_labels = []

        # Add priors for the cartesian state vectors
        for body in np.arange(self.num_secondary_bodies):
            # Get the epoch with the least uncertainty for this body
            # curr_idx = self.body_indices[body_num]
            # radec_uncerts = self.data_table['quant1_err'][curr_idx] + self.data_table['quant2_err'][curr_idx]
            # min_uncert = np.where(radec_uncerts == np.amin(radec_uncerts))[0]
            # best_idx = curr_idx[0][min_uncert[0]]
            datapoints_to_take = 3
            best_idx = self.best_epoch_idx[body]
            best_epochs = self.epochs[best_idx:(best_idx+datapoints_to_take)] # 0 is best, the others are for fitting velocity

            # Get data near best epoch ASSUMING THE BEST IS NOT ONE OF THE LAST TWO EPOCHS OF A GIVEN BODY,
            # also assuming this is in radec
            best_ras = self.input_table['quant1'][best_idx:(best_idx+datapoints_to_take)].copy()
            best_ras_err = self.input_table['quant1_err'][best_idx:(best_idx+datapoints_to_take)].copy()
            best_decs = self.input_table['quant2'][best_idx:(best_idx+datapoints_to_take)].copy()
            best_decs_err = self.input_table['quant2_err'][best_idx:(best_idx+datapoints_to_take)].copy()

            # Convert to AU for prior limits
            best_xs = best_ras / self.plx 
            best_ys = best_decs / self.plx 
            best_xs_err = np.sqrt((best_ras_err / best_ras)**2 + (self.plx_err / self.plx)**2)*np.absolute(best_xs)
            best_ys_err = np.sqrt((best_decs_err / best_decs)**2 + (self.plx_err / self.plx)**2)*np.absolute(best_ys)

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
            mu = consts.G * self.stellar_mass * u.Msun

            mu_vel = 2 * mu / ((x_vel**2 + y_vel**2) * (u.km / u.s * u.km / u.s))
            z_bound = (np.sqrt(mu_vel**2 - (best_xs[0]**2 + best_ys[0]**2)*u.AU *u.AU)).to(u.AU)
            z_bound = z_bound.value

            mu_pos = 2 * mu / np.sqrt((best_xs[0]**2 + best_ys[0]**2) * (u.AU *u.AU))
            z_vel_bound = (np.sqrt(mu_pos - (x_vel**2 + y_vel**2)*(u.km / u.s * u.km / u.s))).to(u.km / u.s)
            z_vel_bound = z_vel_bound.value

            # Add x-coordinate prior
            num_uncerts_x = 5
            basis_priors.append(priors.UniformPrior(best_xs[0] - num_uncerts_x*best_xs_err[0], best_xs[0] + num_uncerts_x*best_xs_err[0]))
            basis_labels.append('x{}'.format(body+1))
            
            # Add y-coordinate prior
            num_uncerts_y = 5
            basis_priors.append(priors.UniformPrior(best_ys[0] - num_uncerts_y*best_ys_err[0], best_ys[0] + num_uncerts_y*best_ys_err[0]))
            basis_labels.append('y{}'.format(body+1))

            # Add z-coordinate prior
            # self.sys_priors.append(priors.UniformPrior(-z_bound,z_bound))
            # self.sys_priors.append(priors.LogUniformPrior(0.0001,z_bound))
            basis_priors.append(priors.GaussianPrior(0.,z_bound / 4, no_negatives=False))
            basis_labels.append('z{}'.format(body+1))

            # Add x-velocity prior
            num_uncerts_xvel = 5
            basis_priors.append(priors.UniformPrior(x_vel - num_uncerts_xvel*x_vel_err, x_vel + num_uncerts_xvel*x_vel_err))
            basis_labels.append('xdot{}'.format(body+1))

            # Add y-velocity prior
            num_uncerts_yvel = 5
            basis_priors.append(priors.UniformPrior(y_vel - num_uncerts_yvel*y_vel_err, y_vel + num_uncerts_yvel*y_vel_err))
            basis_labels.append('ydot{}'.format(body+1))

            # Add z-velocity prior
            # self.sys_priors.append(priors.UniformPrior(-z_vel_bound,z_vel_bound))
            # self.sys_priors.append(priors.LogUniformPrior(0.0001,z_vel_bound))
            basis_priors.append(priors.GaussianPrior(0.,z_vel_bound / 4, no_negatives=False))
            basis_labels.append('zdot{}'.format(body+1))

        # Add parallax prior
        basis_labels.append('plx')
        if self.plx_err > 0:
            basis_priors.append(priors.GaussianPrior(self.plx, self.plx_err))
        else:
            basis_priors.append(self.plx)

        # Add hip_iad priors if necessary
        if self.hipparcos_IAD is not None:
            self.set_hip_iad_priors()

        # Add rv priors
        if self.rv:
            self.set_rv_priors(basis_priors, basis_labels)

        # Add mass priors
        self.set_default_mass_priors(basis_priors, basis_labels)

        return basis_priors, basis_labels

    def to_standard_basis(self, param_arr):
        '''
        Makes a call to 'xyz_to_standard' to convert each companion's xyz parameters
        to the standard parameters an returns the updated array for conversion.

        This conversion does not support OFTI yet.

        Return:
            np.array: Orbital elements in the usual basis for all companions.
        '''

        for body in np.arange(self.num_secondary_bodies)+1:
            startindex = 6 * (body - 1)
            best_idx = self.best_epoch_idx[body - 1]
            constrained_epoch = self.epochs[best_idx]
            mtot = param_arr[-1]

            # Total mass is the sum of companion and stellar
            if self.fit_secondary_mass:
                secondary_m = param_arr[-1-self.num_secondary_bodies+(body-1)]
                mtot = mtot + secondary_m

            to_convert = np.array([*param_arr[startindex:(startindex+6)], param_arr[6 * self.num_secondary_bodies], mtot])
            standard_params = self.xyz_to_standard(constrained_epoch, to_convert)

            # Update param_arr to hold standard parameters
            param_arr[startindex:(startindex+6)] = standard_params[0:6]

        return param_arr

    def xyz_to_standard(self, epoch, elems, tau_ref_epoch=58849):
        """
        Converts array of orbital elements in terms of position and velocity in xyz to the regular one

        Args:
            epoch (float): Date in MJD of observation to calculate time of periastron passage (tau).
            elems (np.array of floats): Orbital elements in xyz basis (x-coordinate [au], y-coordinate [au],
                z-coordinate [au], velocity in x [km/s], velocity in y [km/s], velocity in z [km/s], parallax [mas], total mass of the two-body orbit
                (M_* + M_planet) [Solar masses]). If more than 1 set of parameters is passed, the first dimension must be
                the number of orbital parameter sets, and the second the orbital elements.

        Return:
            np.array: Orbital elements in the usual basis (sma, ecc, inc, aop, pan, tau, plx, mtot)
        """

        if elems.ndim == 1:
            elems = elems[np.newaxis, :]
        # Velocities and positions, with units
        vel = elems[:,3:6] * u.km / u.s # velocities in km / s ?
        pos = elems[:,0:3] * u.AU # positions in AU ?
        vel_magnitude = np.linalg.norm(vel, axis=1)
        pos_magnitude = np.linalg.norm(pos, axis=1)

        # Mass
        mtot = elems[:,7]*u.Msun
        mu = consts.G * mtot # G in m3 kg-1 s-2, mtot in msun

        # Angular momentum, making sure nodal vector is not exactly zero
        h = (np.cross(pos, vel, axis=1)).si
        # if h[0].value == 0.0 and h[1].value == 0.0:
        #     pos[2] = 1e-8*u.AU
        #     h = (np.cross(pos, vel)).si
        h_magnitude = np.linalg.norm(h, axis=1)

        sma = 1 / (2.0 / pos_magnitude - (vel_magnitude**2)/mu)
        sma = sma.to(u.AU)

        ecc = (np.sqrt(1 - h_magnitude**2 / (sma * mu))).value
        e_vector = (np.cross(vel, h, axis=1) / mu[:, None] - pos / pos_magnitude[:, None]).si
        e_vec_magnitude = np.linalg.norm(e_vector, axis=1)

        unit_k = np.array((0,0,1))
        cos_inc = (np.dot(h, unit_k) / h_magnitude).value
        inc = np.arccos(-cos_inc)

        #Nodal vector
        n = np.cross(unit_k, h)
        n_magnitude = np.linalg.norm(n, axis=1)

        # Position angle of the nodes, checking for the right quadrant
        # np.arccos yields angles in [0, pi]
        unit_i = np.array((1,0,0))
        unit_j = np.array((0,1,0))
        cos_pan = (np.dot(n, unit_j) / n_magnitude).value
        pan = np.arccos(cos_pan)
        n_x = np.dot(n, unit_i)
        pan[n_x < 0.0] = 2*np.pi - pan[n_x < 0.0]

        # Argument of periastron, checking for the right quadrant
        cos_aop = (np.sum(n*e_vector, axis=1) / (n_magnitude * e_vec_magnitude)).value
        aop = np.arccos(cos_aop)
        e_vector_z = np.dot(e_vector, unit_k)
        aop[e_vector_z < 0.0] = 2.0*np.pi - aop[e_vector_z < 0.0]

        # True anomaly, checking for the right quadrant
        cos_tanom = (np.sum(pos*e_vector, axis=1) / (pos_magnitude*e_vec_magnitude)).value
        tanom = np.arccos(cos_tanom)
        # Check for places where tanom is nan, due to cos_tanom=1. (for some reason that was a problem)
        tanom = np.where((0.9999<cos_tanom ) & (cos_tanom<1.001), 0.0, tanom)
        rdotv = np.sum(pos*vel, axis=1)
        tanom[rdotv < 0.0] = 2*np.pi - tanom[rdotv < 0.0]

        # Eccentric anomaly to get tau, checking for the right quadrant
        cos_eanom = ((1 - pos_magnitude / sma) / ecc).value
        eanom = np.arccos(cos_eanom)
        # Check for places where eanom is nan, due to cos_eanom = 1.(same problem as above)
        eanom = np.where((0.9999<cos_eanom ) & (cos_eanom<1.001), 0.0, eanom)
        eanom[tanom > np.pi] =  2*np.pi - eanom[tanom > np.pi]

        # Time of periastron passage, using Kepler's equation, in MJD:
        time_tau = epoch - ((np.sqrt(sma**3 / mu)).to(u.day)).value * (eanom - ecc*np.sin(eanom))

        # Calculate period from Kepler's third law, in days:
        period = np.sqrt(4*np.pi**2.0*(sma)**3/mu)
        period = period.to(u.day).value

        # Finally, tau
        tau = (time_tau - tau_ref_epoch) / period
        tau = tau%1.0

        mtot = mtot.value
        sma = sma.value

        results = np.array((sma, ecc, inc, aop, pan, tau, elems[:,6], mtot)).T

        return np.squeeze(results)

    def standard_to_xyz(epoch, elems, tau_ref_epoch=58849, tolerance=1e-9, max_iter=100):
        """
        Converts array of orbital elements from the regular base of Keplerian orbits to positions and velocities in xyz
        Uses code from orbitize.kepler

        Args:
            epoch (float): Date in MJD of observation to calculate time of periastron passage (tau).
            elems (np.array of floats): Orbital elements (sma, ecc, inc, aop, pan, tau, plx, mtot).
                    If more than 1 set of parameters is passed, the first dimension must be
                    the number of orbital parameter sets, and the second the orbital elements.

        Return:
            np.array: Orbital elements in xyz (x-coordinate [au], y-coordinate [au], z-coordinate [au], 
            velocity in x [km/s], velocity in y [km/s], velocity in z [km/s], parallax [mas], total mass of the two-body orbit
                (M_* + M_planet) [Solar masses])
        """

        # Use classical elements to obtain position and velocity in the perifocal coordinate system
        # Then transform coordinates using matrix multiplication

        if elems.ndim == 1:
            elems = elems[np.newaxis, :]
        # Define variables
        sma = elems[:,0] # AU
        ecc = elems[:,1] # [0.0, 1.0]
        inc = elems[:,2] # rad [0, pi]
        aop = elems[:,3] # rad [0, 2 pi]
        pan = elems[:,4] # rad [0, 2 pi]
        tau = elems[:,5] # [0.0, 1.0]
        mtot = elems[:,7] # Msun

        # Just in case so nothing breaks
        ecc = np.where(ecc == 0.0, 1e-8, ecc)
        inc = np.where(inc == 0.0, 1e-8, inc)

        # Begin by calculating the eccentric anomaly
        period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
        period = period.to(u.day).value # Period in days
        mean_motion = 2*np.pi/(period)

        # Mean anomaly
        manom = (mean_motion*(epoch - tau_ref_epoch) - 2*np.pi*tau) % (2.0*np.pi)
        # Eccentric anomaly
        eanom = kepler._calc_ecc_anom(manom, ecc, tolerance=tolerance, max_iter=max_iter)
        # if eanom.ndim == 1:
        #     eanom = eanom[np.newaxis, :]
        # Magnitude of angular momentum:
        h = np.sqrt(consts.G*(mtot*u.Msun)*(sma*u.AU)*(1 - ecc**2))

        # Position vector in the perifocal system in AU
        pos_peri_x = (sma*(np.cos(eanom) - ecc))
        pos_peri_y = (sma*np.sqrt(1 - ecc**2)*np.sin(eanom))
        pos_peri_z = np.zeros(len(elems))

        pos = np.stack((pos_peri_x, pos_peri_y, pos_peri_z)).T
        pos_magnitude = np.linalg.norm(pos, axis=1)

        # Velocity vector in the perifocal system in km/s
        vel_peri_x = - ((np.sqrt(consts.G*(mtot*u.Msun)*(sma*u.AU))*np.sin(eanom) / (pos_magnitude*u.AU)).to(u.km / u.s)).value 
        vel_peri_y = ((h* np.cos(eanom) / (pos_magnitude*u.AU)).to(u.km / u.s)).value
        vel_peri_z = np.zeros(len(elems))

        vel = np.stack((vel_peri_x, vel_peri_y, vel_peri_z)).T

        # Transformation matrix to inertial xyz system, component by component
        pan = pan +np.pi / 2.0
        T_11 = np.cos(pan)*np.cos(aop) - np.sin(pan)*np.sin(aop)*np.cos(inc)
        T_12 = - np.cos(pan)*np.sin(aop) - np.sin(pan)*np.cos(aop)*np.cos(inc)
        T_13 = np.sin(pan)*np.sin(inc)

        T_21 = np.sin(pan)*np.cos(aop) + np.cos(pan)*np.sin(aop)*np.cos(inc)
        T_22 = - np.sin(pan)*np.sin(aop) + np.cos(pan)*np.cos(aop)*np.cos(inc)
        T_23 = - np.cos(pan)*np.sin(inc)

        T_31 = np.sin(aop)*np.sin(inc)
        T_32 = np.cos(aop)*np.sin(inc)
        T_33 = np.cos(inc)

        T = np.array([[T_11, T_12, T_13],
                      [T_21, T_22, T_23],
                      [T_31, T_32, T_33]])

        pos_xyz = np.zeros((len(elems), 3))
        vel_xyz = np.zeros((len(elems), 3))
        for k in range(len(elems)):
            pos_xyz[k,:] =  np.matmul(T[:,:,k], pos[k])
            vel_xyz[k,:] =  np.matmul(T[:,:,k], vel[k])

        # Flipping x-axis sign to increase X as RA increases
        result = np.stack([-pos_xyz[:,0], pos_xyz[:,1], pos_xyz[:,2], -vel_xyz[:,0], vel_xyz[:,1], vel_xyz[:,2], elems[:,6], mtot]).T

        return np.squeeze(result)

# Other conversions
def tau_to_t0(tau, ref_epoch, period, after_date=None):
    """
    DEPRECATING!! Repalced by tau_to_tp
    """
    warnings.warn('DEPRECATION: tau_to_t0 is being deprecated in the next orbitize! release. Please use tau_to_tp instead!', FutureWarning)
    return tau_to_tp(tau, ref_epoch, period, after_date=after_date)

def t0_to_tau(tp, ref_epoch, period):
    """
    DEPRECATING!! Repalced by tp_to_tau
    """
    warnings.warn('DEPRECATION: t0_to_tau is being deprecated in the next orbitize! release. Please use t0_to_tau instead!', FutureWarning)
    return tp_to_tau(tp, ref_epoch, period)

def tau_to_tp(tau, ref_epoch, period, after_date=None):
    """
    Convert tau (epoch of periastron in fractional orbital period after ref epoch) to
    t_p (date in days, usually MJD, but works with whatever system ref_epoch is given in)

    Args:
        tau (float or np.array): value of tau to convert
        ref_epoch (float or np.array): date (in days, typically MJD) that tau is defined relative to
        period (float or np.array): period (in years) that tau is noralized with
        after_date (float): tp will be the first periastron after this date. If None, use ref_epoch.

    Returns:
        tp (float or np.array): corresponding t_p of the taus
    """
    period_days = period * u.year.to(u.day)

    tp = tau * (period_days) + ref_epoch

    if after_date is not None:
        num_periods = (after_date - tp)/period_days
        num_periods = int(np.ceil(num_periods))
        
        tp += num_periods * period_days

    return tp

def tp_to_tau(tp, ref_epoch, period):
    """
    Convert t_p to tau

    Args:
        tp (float or np.array): value to t_p to convert (days, typically MJD)
        ref_epoch (float or np.array): reference epoch (in days) that tau is defined from. Same system as tp (e.g., MJD)
        period (float or np.array): period (in years) that tau is defined by

    Returns:
        tau (float or np.array): corresponding taus
    """
    tau = (tp - ref_epoch)/(period * u.year.to(u.day))
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

    tp = tau_to_tp(old_tau, old_epoch, period)
    new_tau = tp_to_tau(tp, new_epoch, period)

    return new_tau

def tau_to_manom(date, sma, mtot, tau, tau_ref_epoch):
    """
    Gets the mean anomlay
    
    Args:
        date (float or np.array): MJD
        sma (float): semi major axis (AU)
        mtot (float): total mass (M_sun)
        tau (float): epoch of periastron, in units of the orbital period
        tau_ref_epoch (float): reference epoch for tau
        
    Returns:
        mean_anom (float or np.array): mean anomaly on that date [0, 2pi)
    """

    period = np.sqrt(
        4 * np.pi**2.0 * (sma * u.AU)**3 /
        (consts.G * (mtot * u.Msun))
    )
    period = period.to(u.day).value

    frac_date = (date - tau_ref_epoch)/period
    frac_date %= 1

    mean_anom = (frac_date - tau) * 2 * np.pi
    mean_anom %= 2 * np.pi

    return mean_anom