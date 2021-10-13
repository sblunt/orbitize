import numpy as np
import astropy.units as u, astropy.constants as consts
import warnings
import abc

from orbitize import priors, kepler
from scipy.optimize import fsolve

class Basis(abc.ABC):
    """
    Abstract base class for different basis sets. All new basis objects should inherit from
    this class. This class is meant to control how priors are assigned to various basis sets
    and how conversions are made from the basis sets to the standard keplarian set.

    Author: Tirth, 2021
    """

    def __init__(
        self, stellar_or_system_mass, mass_err, plx, plx_err, num_secondary_bodies, 
        fit_secondary_mass, angle_upperlim=2*np.pi, hipparcos_IAD=None, 
        rv=False, rv_instruments=None
    ):

        self.stellar_or_system_mass = stellar_or_system_mass
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
        self.default_priors = {
            'sma' : priors.LogUniformPrior(0.001, 1e4), 
            'per' : priors.LogUniformPrior(1e-5, 1e6),
            'ecc' : priors.UniformPrior(0., 1.), 
            'inc' : priors.SinPrior(), 
            'aop' : priors.UniformPrior(0., 2.*np.pi),
            'pan' : priors.UniformPrior(0., angle_upperlim), 
            'tau' : priors.UniformPrior(0., 1.),
            'K' : priors.LogUniformPrior(1e-4, 10)
        }
        

    @abc.abstractmethod
    def construct_priors(self):
        pass

    @abc.abstractmethod
    def to_standard_basis(self, param_arr, param_idx):
        pass

    def verify_params(self):
        '''
        Displays warnings about the 'fit_secondary_mass' and 'rv' parameters for 
        all basis sets. Can be overriden by any basis class depending on the 
        necessary parameters that need to be checked. 
        '''
        if not self.fit_secondary_mass and self.rv:
            warnings.warn(
                """"
                Radial velocity data found in input data, but rv parameters will 
                not be sampled. To sample rv parameters, set 'fit_secondary_mass' 
                to True.
                """
            )

    def set_hip_iad_priors(self, priors_arr, labels_arr):
        '''
        Adds the necessary priors relevant to the hipparcos data to 'priors_arr' 
        and updates 'labels_arr' with the priors' corresponding labels.

        Args:
            priors_arr (list of orbitize.priors.Prior objects): holds the prior 
                objects for each parameter to be fitted (updated here)
            labels_arr (list of strings): holds the name of all the parameters 
                to be fitted (updated here)
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
        Adds the necessary priors if radial velocity data is supplied to 
        'priors_arr' and updates 'labels_arr' with the priors' corresponding 
        labels. This function assumes that 'rv' data has been supplied and 
        a secondary mass is being fitted for.

        Args:
            priors_arr (list of orbitize.priors.Prior objects): holds the prior 
                objects for each parameter to be fitted (updated here)
            labels_arr (list of strings): holds the name of all the parameters 
                to be fitted (updated here)
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
            priors_arr (list of orbitize.priors.Prior objects): holds the prior 
                objects for each parameter to be fitted (updated here)
            labels_arr (list of strings): holds the name of all the parameters 
                to be fitted (updated here)
        '''
        if self.fit_secondary_mass:
            for body in np.arange(self.num_secondary_bodies)+1:
                priors_arr.append(priors.LogUniformPrior(1e-6, 2)) # in Solar masses
                labels_arr.append('m{}'.format(body))
            labels_arr.append('m0')
        else:
            labels_arr.append('mtot')

        if self.mass_err > 0:
            priors_arr.append(priors.GaussianPrior(self.stellar_or_system_mass, self.mass_err))
        else:
            priors_arr.append(self.stellar_or_system_mass)


class Standard(Basis):
    '''
    Standard basis set based upon the 6 standard Keplarian elements: (sma, ecc, inc, aop, pan, tau).

    Args:
        stellar_or_system_mass (float): mass of the primary star (if fitting for
            dynamical masses of both components) or total system mass (if
            fitting using relative astrometry only) [M_sol]
        mass_err (float): uncertainty on 'stellar_or_system_mass', in M_sol
        plx (float): mean parallax of the system, in mas
        plx_err (float): uncertainty on 'plx', in mas
        num_secondary_bodies (int): number of secondary bodies in the system, should be at least 1
        fit_secondary_mass (bool): if True, include the dynamical mass of orbitting body as fitted parameter, if False,
            'stellar_or_system_mass' is taken to be total mass
        angle_upperlim (float): either pi or 2pi, to restrict the prior range for 'pan' parameter (default: 2pi)
        hipparcos_IAD (orbitize.HipparcosLogProb object): if not 'None', then add relevant priors to this data (default: None)
        rv (bool): if True, then there is radial velocity data and assign radial velocity priors, if False, then there
            is no radial velocity data and radial velocity priors are not assigned (default: False)
        rv_instruments (np.array): array of unique rv instruments from the originally supplied data (default: None)
    '''

    def __init__(self, stellar_or_system_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
        angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, rv_instruments=None):

        super(Standard, self).__init__(stellar_or_system_mass, mass_err, plx, plx_err, num_secondary_bodies, 
            fit_secondary_mass, angle_upperlim, hipparcos_IAD, rv, rv_instruments)

    def construct_priors(self):
        '''
        Generates the parameter label array and initializes the corresponding priors for each
        parameter that's to be sampled. For the standard basis, the parameters common to each
        companion are: sma, ecc, inc, aop, pan, tau. Parallax, hipparcos (optional), rv (optional),
        and mass priors are added at the end.

        Returns:
            tuple:

                list: list of strings (labels) that indicate the names of each parameter to sample

                list: list of orbitize.priors.Prior objects that indicate the prior distribution of each label
        '''
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
            self.set_hip_iad_priors(basis_priors, basis_labels)

        # Add rv priors
        if self.rv and self.fit_secondary_mass:
            self.set_rv_priors(basis_priors, basis_labels)

        # Add mass priors
        self.set_default_mass_priors(basis_priors, basis_labels)

        # Define param label dictionary in current basis & standard basis
        self.param_idx = dict(zip(basis_labels, np.arange(len(basis_labels))))
        self.standard_basis_idx = dict(zip(basis_labels, np.arange(len(basis_labels))))

        return basis_priors, basis_labels

    def to_standard_basis(self, param_arr):
        '''
        For standard basis, no conversion needs to be made.

        Args:
            param_arr (np.array of float): RxM array of fitting parameters in the standard basis, 
                where R is the number of parameters being fit, and M is the number of orbits. If 
                M=1 (for MCMC), this can be a 1d array.

        Returns: 
            np.array of float: ``param_arr`` without any modification
        '''
        return param_arr

class Period(Basis):
    '''
    Modification of the standard basis, swapping our sma for period: (per, ecc, inc, aop, pan, tau).

    Args:
        stellar_or_system_mass (float): mass of the primary star (if fitting for
            dynamical masses of both components) or total system mass (if
            fitting using relative astrometry only) [M_sol]
        mass_err (float): uncertainty on 'stellar_or_system_mass', in M_sol
        plx (float): mean parallax of the system, in mas
        plx_err (float): uncertainty on 'plx', in mas
        num_secondary_bodies (int): number of secondary bodies in the system, should be at least 1
        fit_secondary_mass (bool): if True, include the dynamical mass of orbitting body as fitted parameter, if False,
            'stellar_or_system_mass' is taken to be total mass
        angle_upperlim (float): either pi or 2pi, to restrict the prior range for 'pan' parameter (default: 2pi)
        hipparcos_IAD (orbitize.HipparcosLogProb object): if not 'None', then add relevant priors to this data (default: None)
        rv (bool): if True, then there is radial velocity data and assign radial velocity priors, if False, then there
            is no radial velocity data and radial velocity priors are not assigned (default: False)
        rv_instruments (np.array): array of unique rv instruments from the originally supplied data (default: None)
    '''

    def __init__(self, stellar_or_system_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
        angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, rv_instruments=None):

        super(Period, self).__init__(stellar_or_system_mass, mass_err, plx, plx_err, num_secondary_bodies, 
            fit_secondary_mass, angle_upperlim, hipparcos_IAD, rv, rv_instruments)

    def construct_priors(self):
        '''
        Generates the parameter label array and initializes the corresponding priors for each
        parameter that's to be sampled. For the standard basis, the parameters common to each
        companion are: per, ecc, inc, aop, pan, tau. Parallax, hipparcos (optional), rv (optional),
        and mass priors are added at the end.

        Returns:
            tuple:

                list: list of strings (labels) that indicate the names of each parameter to sample

                list: list of orbitize.priors.Prior objects that indicate the prior distribution of each label
        '''
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
            self.set_hip_iad_priors(basis_priors, basis_labels)

        # Add rv priors
        if self.rv and self.fit_secondary_mass:
            self.set_rv_priors(basis_priors, basis_labels)

        # Add mass priors
        self.set_default_mass_priors(basis_priors, basis_labels)

        # Define param label dictionary in current basis & standard basis
        self.param_idx = dict(zip(basis_labels, np.arange(len(basis_labels))))
        self.standard_basis_idx = dict(zip(basis_labels, np.arange(len(basis_labels))))

        for body_num in np.arange(self.num_secondary_bodies) + 1:
            self.standard_basis_idx['sma{}'.format(body_num)] = self.param_idx['per{}'.format(body_num)]

        return basis_priors, basis_labels

    def to_standard_basis(self, param_arr):
        '''
        Convert parameter array from period basis to standard basis by swapping out the period
        parameter to semi-major axis for each companion.
        
        Args:
            param_arr (np.array of float): RxM array of fitting parameters in the period basis, 
                where R is the number of parameters being fit, and M is the number of orbits. If 
                M=1 (for MCMC), this can be a 1D array.

        Returns:
            np.array of float: modifies 'param_arr' to contain the semi-major axis for each companion
                in each orbit rather than period. Shape of 'param_arr' remains the same.
        '''
        for body_num in np.arange(self.num_secondary_bodies)+1:
            per = param_arr[self.param_idx['per{}'.format(body_num)]]

            if self.fit_secondary_mass:
                # Assume two-body system
                m_secondary = param_arr[self.param_idx['m{}'.format(body_num)]]
                m0 = param_arr[self.param_idx['m0']]
                mtot = m_secondary + m0
            else:
                mtot = param_arr[self.param_idx['mtot']]

            # Compute semi-major axis using Kepler's Third Law and replace period
            sma = np.cbrt((consts.G*(mtot*u.Msun)*((per*u.year)**2))/(4*np.pi**2))
            sma = sma.to(u.AU).value
            param_arr[self.standard_basis_idx['per{}'.format(body_num)]] = sma

        return param_arr

    def to_period_basis(self, param_arr):
        '''
        Convert parameter array from standard basis to period basis by swapping out the semi-major
        axis parameter to period for each companion. This function is used primarily for testing
        purposes.
        
        Args:
            param_arr (np.array of float): RxM array of fitting parameters in the standard basis, 
                where R is the number of parameters being fit, and M is the number of orbits. If 
                M=1 (for MCMC), this can be a 1D array.

        Returns:
            np.array of float: modifies 'param_arr' to contain the period for each companion
                in each orbit rather than semi-major axis. Shape of 'param_arr' remains the same.
        '''
        for body_num in np.arange(self.num_secondary_bodies)+1:
            sma = param_arr[self.standard_basis_idx['sma{}'.format(body_num)]]

            if self.fit_secondary_mass:
                # Assume two-body system
                m_secondary = param_arr[self.standard_basis_idx['m{}'.format(body_num)]]
                m0 = param_arr[self.standard_basis_idx['m0']]
                mtot = m_secondary + m0
            else:
                mtot = param_arr[self.standard_basis_idx['mtot']]

            per = np.sqrt((4*(np.pi**2)*(sma*u.AU)**3) / (consts.G*(mtot*u.Msun)))
            per = per.to(u.year).value
            param_arr[self.param_idx['per{}'.format(body_num)]] = per
        
        return param_arr

class SemiAmp(Basis):
    '''
    Modification of the standard basis, swapping our sma for period and additionally sampling in
    the stellar radial velocity semi-amplitude: (per, ecc, inc, aop, pan, tau, K).

    .. Note:: Ideally, 'fit_secondary_mass' is true and rv data is supplied.

    Args:
        stellar_or_system_mass (float): mass of the primary star (if fitting for
            dynamical masses of both components) or total system mass (if
            fitting using relative astrometry only) [M_sol]
        mass_err (float): uncertainty on 'stellar_or_system_mass', in M_sol
        plx (float): mean parallax of the system, in mas
        plx_err (float): uncertainty on 'plx', in mas
        num_secondary_bodies (int): number of secondary bodies in the system, should be at least 1
        fit_secondary_mass (bool): if True, include the dynamical mass of orbitting body as fitted parameter, if False,
            'stellar_or_system_mass' is taken to be total mass
        angle_upperlim (float): either pi or 2pi, to restrict the prior range for 'pan' parameter (default: 2*pi)
        hipparcos_IAD (orbitize.HipparcosLogProb object): if not 'None', then add relevant priors to this data (default: None)
        rv (bool): if True, then there is radial velocity data and assign radial velocity priors, if False, then there
            is no radial velocity data and radial velocity priors are not assigned (default: False)
        rv_instruments (np.array): array of unique rv instruments from the originally supplied data (default: None)
    '''

    def __init__(self, stellar_or_system_mass, mass_err, plx, plx_err, num_secondary_bodies, fit_secondary_mass, 
        angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, rv_instruments=None):

        super(SemiAmp, self).__init__(stellar_or_system_mass, mass_err, plx, plx_err, num_secondary_bodies, 
            fit_secondary_mass, angle_upperlim, hipparcos_IAD, rv, rv_instruments)

    def construct_priors(self):
        '''
        Generates the parameter label array and initializes the corresponding priors for each
        parameter that's to be sampled. For the semi-amp basis, the parameters common to each
        companion are: per, ecc, inc, aop, pan, tau, K (stellar rv semi-amplitude). Parallax, 
        hipparcos (optional), rv (optional), and mass priors are added at the end.

        The mass parameter will always be m0.

        Returns:
            tuple:

            list: list of strings (labels) that indicate the names of each parameter to sample
            
            list: list of orbitize.priors.Prior objects that indicate the prior distribution of each label
        '''
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
            self.set_hip_iad_priors(basis_priors, basis_labels)

        # Add rv priors
        if self.rv and self.fit_secondary_mass:
            self.set_rv_priors(basis_priors, basis_labels)

        # Add star mass prior (for now, regardless of whether 'fit_secondary_mass' is true)
        if self.mass_err > 0:
            basis_priors.append(priors.GaussianPrior(self.stellar_or_system_mass, self.mass_err))
        else:
            basis_priors.append(self.stellar_or_system_mass)

        basis_labels.append('m0')

        # Define param label dictionary in current basis & standard basis
        self.param_idx = dict(zip(basis_labels, np.arange(len(basis_labels))))
        self.standard_basis_idx = dict(zip(basis_labels, np.arange(len(basis_labels))))

        for body_num in np.arange(self.num_secondary_bodies) + 1:
            self.standard_basis_idx['sma{}'.format(body_num)] = self.param_idx['per{}'.format(body_num)]
            self.standard_basis_idx['m{}'.format(body_num)] = self.param_idx['K{}'.format(body_num)]

        return basis_priors, basis_labels

    def to_standard_basis(self, param_arr):
        '''
        Convert parameter array from semi-amp basis to standard basis by swapping out the period
        parameter to semi-major axis for each companion and computing the masses of each
        companion.
        
        Args:
            param_arr (np.array of float): RxM array of fitting parameters in the period basis, 
                where R is the number of parameters being fit, and M is the number of orbits. If 
                M=1 (for MCMC), this can be a 1D array.

        Returns:
            np.array of float: modifies 'param_arr' to contain the semi-major axis for each companion
                in each orbit rather than period, removes stellar rv semi-amplitude parameters for
                each companion, and appends the companion masses to 'param_arr'
        '''
        m0 = param_arr[self.param_idx['m0']]

        # Compute each companion's mass and sma
        for body_num in np.arange(self.num_secondary_bodies) + 1:
            period = param_arr[self.param_idx['per{}'.format(body_num)]]
            ecc = param_arr[self.param_idx['ecc{}'.format(body_num)]]
            inc = param_arr[self.param_idx['inc{}'.format(body_num)]]
            semi_amp = param_arr[self.param_idx['K{}'.format(body_num)]]

            # Replace semi-amp with companion mass and period with sma
            companion_m = self.compute_companion_mass(period, ecc, inc, semi_amp, m0)
            param_arr[self.standard_basis_idx['m{}'.format(body_num)]] = companion_m
            companion_sma = self.compute_companion_sma(period, m0, companion_m)
            param_arr[self.standard_basis_idx['sma{}'.format(body_num)]] = companion_sma

        return param_arr

    def func(self, x, lhs, m0):
        '''
        Define function for scipy.fsolve to use when computing companion mass.

        Args:
            x (float): the companion mass to be calculated (Msol)
            lhs (float): the left hand side of the rv semi-amplitude equation (Msol^(1/3))
            m0 (float): the stellar mass (Msol)

        Returns:
            float: the difference between the rhs and lhs of the rv semi-amplitude equation, 'x' is a
                good companion mass when this difference is very close to zero
        '''
        return ((x / ((x + m0)**(2/3))) - lhs)

    def compute_companion_mass(self, period, ecc, inc, semi_amp, m0):
        '''
        Computes a single companion's mass given period, eccentricity, inclination, stellar rv semi-amplitude,
        and stellar mass. Uses scipy.fsolve to compute the masses numerically.

        Args:
            period (np.array of float): the period values for each orbit for a single companion (can be float)
            ecc (np.array of float): the eccentricity values for each orbit for a single companion (can be float)
            inc (np.array of float): the inclination values for each orbit for a single companion (can be float)
            semi_amp (np.array of float): the stellar rv-semi amp values for each orbit (can be float)
            m0 (np.array of float): the stellar mass for each orbit (can be float)

        Returns:
            np.array of float: the companion mass values for each orbit (can also just be a single float)
        '''

        # Define LHS of equation
        kms = u.km / u.s
        lhs = ((semi_amp*kms)*((1-ecc**2)**(1/2))*((period*u.yr)**(1/3))*(consts.G**(-1/3))*((4*np.pi**2)**(-1/6))) / (np.sin(inc))
        lhs = (lhs.to((u.solMass)**(1/3))).value

        m_n = []

        # Solve for companion mass numerically, making initial guess at center of uniform prior distribution (Msol)
        if (not hasattr(m0, '__len__')):
            comp_mass = fsolve(self.func, x0=1e-3, args=(lhs, m0))
            m_n.append(comp_mass[0])
        else:
            for orbit in range(len(m0)):
                comp_mass = fsolve(self.func, x0=1e-3, args=(lhs[orbit], m0[orbit]))
                m_n.append(comp_mass[0])

        # squash dimensions
        if len(m_n) == 1:
            m_n = m_n[0]

        return m_n

    def compute_companion_sma(self, period, m0, m_n):
        '''
        Computes a single companion's semi-major axis using Kepler's Third Law for each orbit.

        Args:
            period (np.array of float): the period values for each orbit for a single companion (can be float)
            m0 (np.array of float): the stellar mass for each orbit (can be float)
            m_n (np.array of float): the companion mass for each orbit (can be float)

        Returns:
            np.array of float: the semi-major axis values for each orbit                        
        '''
        sma = np.cbrt((consts.G*((m0+m_n)*u.Msun)*((period*u.yr)**2))/(4*np.pi**2))
        sma = sma.to(u.AU).value

        return sma

    def to_semi_amp_basis(self, param_arr):
        '''
        Convert parameter array from standard basis to semi-amp basis by swapping out the
        semi-major axis parameter to period for each companion and computing the stellar
        rv semi-amplitudes for each companion.
        
        Args:
            param_arr (np.array of float): RxM array of fitting parameters in the period basis, 
                where R is the number of parameters being fit, and M is the number of orbits. If 
                M=1 (for MCMC), this can be a 1D array.

        Returns:
            np.array of float: modifies 'param_arr' to contain the semi-major axis for each companion
                in each orbit rather than period, appends stellar rv semi-amplitude parameters, and
                removes companion masses
        '''

        for body_num in np.arange(self.num_secondary_bodies) + 1:

            # Grab necessary parameters for conversion
            sma = param_arr[self.standard_basis_idx['sma{}'.format(body_num)]]
            ecc = param_arr[self.standard_basis_idx['ecc{}'.format(body_num)]]
            inc = param_arr[self.standard_basis_idx['inc{}'.format(body_num)]]
            m_n = param_arr[self.standard_basis_idx['m{}'.format(body_num)]]
            m0 = param_arr[self.standard_basis_idx['m0']]
            mtot = m_n + m0

            # Get stellar semi-amplitude
            K_n = (np.sqrt(consts.G / (1 - ecc**2)))*(m_n*u.Msun)*(np.sin(inc))*((mtot*u.Msun)**(-1/2))*((sma*u.AU)**(-1/2))
            kms = u.km / u.s
            K_n = K_n.to(kms).value

            # Compute Period replace in array
            per = np.sqrt((4*(np.pi**2)*(sma*u.AU)**3) / (consts.G*(mtot*u.Msun)))
            per = per.to(u.year).value
            param_arr[self.param_idx['per{}'.format(body_num)]] = per

            # Replace companion mass with semi-amplitude
            param_arr[self.param_idx['K{}'.format(body_num)]] = K_n

        return param_arr

    def verify_params(self):
        '''
        Additionally warns that this basis will sample stellar mass rather than sample mass
        regardless of whether 'fit_secondary_mass' is True or not.
        '''
        super(SemiAmp, self).verify_params()

        if not self.fit_secondary_mass:
            warnings.warn("This basis will not sample total mass. It will sample stellar mass.")

class XYZ(Basis):
    '''
    Defines an orbit using the companion's position and velocity components in XYZ space (x, y, z, xdot, ydot, zdot).
    The conversion algorithms used for this basis are defined in the following paper:
    http://www.dept.aoe.vt.edu/~lutze/AOE4134/9OrbitInSpace.pdf

    .. Note:: Does not have support with sep,pa data yet.

    .. Note:: Does not work for all multi-body data.

    Args:
        stellar_or_system_mass (float): mass of the primary star (if fitting for
            dynamical masses of both components) or total system mass (if
            fitting using relative astrometry only) [M_sol]
        mass_err (float): uncertainty on 'stellar_or_system_mass', in M_sol
        plx (float): mean parallax of the system, in mas
        plx_err (float): uncertainty on 'plx', in mas
        num_secondary_bodies (int): number of secondary bodies in the system, should be at least 1
        fit_secondary_mass (bool): if True, include the dynamical mass of orbitting body as fitted parameter, if False,
            'stellar_or_system_mass' is taken to be total mass
        input_table (astropy.table.Table): output from 'orbitize.read_input.read_file()'
        best_epoch_idx (list): indices of the epochs corresponding to the smallest uncertainties
        epochs (list): all of the astrometric epochs from 'input_table'
        angle_upperlim (float): either pi or 2pi, to restrict the prior range for 'pan' parameter (default: 2*pi)
        hipparcos_IAD (orbitize.HipparcosLogProb object): if not 'None', then add relevant priors to this data (default: None)
        rv (bool): if True, then there is radial velocity data and assign radial velocity priors, if False, then there
            is no radial velocity data and radial velocity priors are not assigned (default: False)
        rv_instruments (np.array): array of unique rv instruments from the originally supplied data (default: None)

    Author: Rodrigo
    '''
    def __init__(
        self, stellar_or_system_mass, mass_err, plx, plx_err, num_secondary_bodies, 
        fit_secondary_mass, data_table, best_epoch_idx, epochs, 
        angle_upperlim=2*np.pi, hipparcos_IAD=None, rv=False, 
        rv_instruments=None
    ):

        super(XYZ, self).__init__(stellar_or_system_mass, mass_err, plx, plx_err, 
            num_secondary_bodies, fit_secondary_mass, angle_upperlim, 
            hipparcos_IAD, rv, rv_instruments
        )

        self.data_table = data_table
        self.best_epoch_idx = best_epoch_idx
        self.epochs = epochs

    def construct_priors(self):
        '''
        Generates the parameter label array and initializes the corresponding priors for each
        parameter that's to be sampled. For the xyz basis, the parameters common to each
        companion are: x, y, z, xdot, ydot, zdot. Parallax, hipparcos (optional), rv (optional),
        and mass priors are added at the end.

        The xyz basis describes the position and velocity vectors with reference to the local coordinate 
        system (the origin of the system is star).

        Returns:
            tuple:

                list: list of strings (labels) that indicate the names of each parameter to sample

                list: list of orbitize.priors.Prior objects that indicate the prior distribution of each label
        '''

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

            # Get data near best epoch ASSUMING THE BEST IS NOT ONE OF THE LAST TWO EPOCHS OF A GIVEN BODY 
            # also assuming this is in radec
            best_ras = self.data_table['quant1'][best_idx:(best_idx+datapoints_to_take)].copy()
            best_ras_err = self.data_table['quant1_err'][best_idx:(best_idx+datapoints_to_take)].copy()
            best_decs = self.data_table['quant2'][best_idx:(best_idx+datapoints_to_take)].copy()
            best_decs_err = self.data_table['quant2_err'][best_idx:(best_idx+datapoints_to_take)].copy()

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
            mu = consts.G * self.stellar_or_system_mass * u.Msun

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
            self.set_hip_iad_priors(basis_priors, basis_labels)

        # Add rv priors
        if self.rv and self.fit_secondary_mass:
            self.set_rv_priors(basis_priors, basis_labels)

        # Add mass priors
        self.set_default_mass_priors(basis_priors, basis_labels)

        # Define param label dictionary in current basis & standard basis
        self.param_idx = dict(zip(basis_labels, np.arange(len(basis_labels))))
        self.standard_basis_idx = dict(zip(basis_labels, np.arange(len(basis_labels))))

        for body_num in np.arange(self.num_secondary_bodies) + 1:
            self.standard_basis_idx['sma{}'.format(body_num)] = self.param_idx['x{}'.format(body_num)]
            self.standard_basis_idx['ecc{}'.format(body_num)] = self.param_idx['y{}'.format(body_num)]
            self.standard_basis_idx['inc{}'.format(body_num)] = self.param_idx['z{}'.format(body_num)]
            self.standard_basis_idx['aop{}'.format(body_num)] = self.param_idx['xdot{}'.format(body_num)]
            self.standard_basis_idx['pan{}'.format(body_num)] = self.param_idx['ydot{}'.format(body_num)]
            self.standard_basis_idx['tau{}'.format(body_num)] = self.param_idx['zdot{}'.format(body_num)]

        return basis_priors, basis_labels

    def to_standard_basis(self, param_arr):
        '''
        Makes a call to 'xyz_to_standard' to convert each companion's xyz parameters
        to the standard parameters an returns the updated array for conversion.

        Args:
            param_arr (np.array of float): RxM array of fitting parameters in the period basis, 
                where R is the number of parameters being fit, and M is the number of orbits. If 
                M=1 (for MCMC), this can be a 1D array.

        Return:
            np.array: Orbital elements in the standard basis for all companions.
        '''
        for body_num in np.arange(self.num_secondary_bodies)+1:
            best_idx = self.best_epoch_idx[body_num - 1]
            constrained_epoch = self.epochs[best_idx]

            # Total mass is the sum of companion and stellar
            if self.fit_secondary_mass:
                secondary_m = param_arr[self.param_idx['m{}'.format(body_num)]]
                m0 = param_arr[self.param_idx['m0']]
                mtot = m0 + secondary_m
            else:
                mtot = param_arr[self.param_idx['mtot']]

            to_convert = np.array([
                param_arr[self.param_idx['x{}'.format(body_num)]],
                param_arr[self.param_idx['y{}'.format(body_num)]],
                param_arr[self.param_idx['z{}'.format(body_num)]],
                param_arr[self.param_idx['xdot{}'.format(body_num)]],
                param_arr[self.param_idx['ydot{}'.format(body_num)]],
                param_arr[self.param_idx['zdot{}'.format(body_num)]],
                param_arr[self.param_idx['plx']],
                mtot
            ])
            standard_params = self.xyz_to_standard(constrained_epoch, to_convert)

            # Update param_arr to hold standard parameters
            param_arr[self.standard_basis_idx['sma{}'.format(body_num)]] = standard_params[0]
            param_arr[self.standard_basis_idx['ecc{}'.format(body_num)]] = standard_params[1]
            param_arr[self.standard_basis_idx['inc{}'.format(body_num)]] = standard_params[2]
            param_arr[self.standard_basis_idx['aop{}'.format(body_num)]] = standard_params[3]
            param_arr[self.standard_basis_idx['pan{}'.format(body_num)]] = standard_params[4]
            param_arr[self.standard_basis_idx['tau{}'.format(body_num)]] = standard_params[5]
            param_arr[self.standard_basis_idx['plx']] = standard_params[6]
            param_arr[self.standard_basis_idx['mtot']] = standard_params[7]

        return param_arr

    def xyz_to_standard(self, epoch, elems, tau_ref_epoch=58849):
        """
        Converts array of orbital elements in terms of position and velocity in 
        xyz to the standard basis.

        Args:
            epoch (float): Date in MJD of observation to calculate time of 
                periastron passage (tau).
            elems (np.array of floats): Orbital elements in xyz basis 
                (x-coordinate [au], y-coordinate [au], z-coordinate [au], 
                velocity in x [km/s], velocity in y [km/s], velocity in z [km/s], 
                parallax [mas], total mass of the two-body orbit (M_* + M_planet) 
                [Solar masses]). If more than 1 set of parameters is passed, the 
                first dimension must be the number of orbital parameter sets, 
                and the second the orbital elements.

        Return:
            np.array: Orbital elements in the standard basis 
                (sma, ecc, inc, aop, pan, tau, plx, mtot)
        """
        if elems.ndim == 1:
            elems = elems[:, np.newaxis]
        # Velocities and positions, with units
        vel = elems[3:6, :] * u.km / u.s # velocities in km / s ?
        pos = elems[0:3, :] * u.AU # positions in AU ?
        vel_magnitude = np.linalg.norm(vel, axis=0)
        pos_magnitude = np.linalg.norm(pos, axis=0)

        # Mass
        mtot = elems[7, :]*u.Msun
        mu = consts.G * mtot # G in m3 kg-1 s-2, mtot in msun

        # Angular momentum, making sure nodal vector is not exactly zero
        h = (np.cross(pos, vel, axis=0)).si
        # if h[0].value == 0.0 and h[1].value == 0.0:
        #     pos[2] = 1e-8*u.AU
        #     h = (np.cross(pos, vel)).si
        h_magnitude = np.linalg.norm(h, axis=0)

        sma = 1 / (2.0 / pos_magnitude - (vel_magnitude**2)/mu)
        sma = sma.to(u.AU)

        ecc = (np.sqrt(1 - h_magnitude**2 / (sma * mu))).value
        e_vector = (np.cross(vel, h, axis=0) / mu - pos / pos_magnitude).si
        e_vec_magnitude = np.linalg.norm(e_vector, axis=0)

        unit_k = np.array((0,0,1))[:, None]
        cos_inc = (np.sum(h*unit_k, axis=0) / h_magnitude).value
        inc = np.arccos(-cos_inc) # Take arccos of positive cos_inc?

        #Nodal vector
        n = np.cross(unit_k, h, axis=0)
        n_magnitude = np.linalg.norm(n, axis=0)

        # Position angle of the nodes, checking for the right quadrant
        # np.arccos yields angles in [0, pi]
        unit_i = np.array((1,0,0))[:, None]
        unit_j = np.array((0,1,0))[:, None]
        cos_pan = (np.sum(n*unit_j, axis=0) / n_magnitude).value # take dot product with i?
        pan = np.arccos(cos_pan)
        n_x = np.sum(n*unit_i, axis=0)
        pan[n_x < 0.0] = 2*np.pi - pan[n_x < 0.0]

        # Argument of periastron, checking for the right quadrant
        cos_aop = (np.sum(n*e_vector, axis=0) / (n_magnitude * e_vec_magnitude)).value
        aop = np.arccos(cos_aop)
        e_vector_z = np.sum(e_vector*unit_k, axis=0)
        aop[e_vector_z < 0.0] = 2.0*np.pi - aop[e_vector_z < 0.0]

        # True anomaly, checking for the right quadrant
        cos_tanom = (np.sum(pos*e_vector, axis=0) / (pos_magnitude*e_vec_magnitude)).value
        tanom = np.arccos(cos_tanom)
        # Check for places where tanom is nan, due to cos_tanom=1. (for some reason that was a problem)
        # tanom = np.where((0.9999<cos_tanom) & (cos_tanom<1.001), 0.0, tanom)
        rdotv = np.sum(pos*vel, axis=0)
        tanom[rdotv < 0.0] = 2*np.pi - tanom[rdotv < 0.0]

        # Eccentric anomaly to get tau, checking for the right quadrant
        cos_eanom = ((1 - pos_magnitude / sma) / ecc).value
        eanom = np.arccos(cos_eanom)
        # Check for places where eanom is nan, due to cos_eanom = 1.(same problem as above)
        # eanom = np.where((0.9999<cos_eanom ) & (cos_eanom<1.001), 0.0, eanom)
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

        results = np.stack([sma, ecc, inc, aop, pan, tau, elems[6, :], mtot])
        return np.squeeze(results)

    def to_xyz_basis(self, param_arr):
        '''
        Makes a call to 'standard_to_xyz' to convert each companion's standard keplerian parameters
        to the xyz parameters an returns the updated array for conversion.

        Args:
            param_arr (np.array of float): RxM array of fitting parameters in the period basis, 
                where R is the number of parameters being fit, and M is the number of orbits. If 
                M=1 (for MCMC), this can be a 1D array.

        Return:
            np.array: Orbital elements in the xyz for all companions.
        '''
        for body_num in np.arange(self.num_secondary_bodies)+1:
            best_idx = self.best_epoch_idx[body_num - 1]
            constrained_epoch = self.epochs[best_idx]

            # Get total mass
            if self.fit_secondary_mass:
                secondary_m = param_arr[self.param_idx['m{}'.format(body_num)]]
                m0 = param_arr[self.standard_basis_idx['m0']]
                mtot = m0 + secondary_m
            else:
                mtot = param_arr[self.param_idx['mtot']]

            # Make conversion
            to_convert = np.array([
                param_arr[self.standard_basis_idx['sma{}'.format(body_num)]],
                param_arr[self.standard_basis_idx['ecc{}'.format(body_num)]],
                param_arr[self.standard_basis_idx['inc{}'.format(body_num)]],
                param_arr[self.standard_basis_idx['aop{}'.format(body_num)]],
                param_arr[self.standard_basis_idx['pan{}'.format(body_num)]],
                param_arr[self.standard_basis_idx['tau{}'.format(body_num)]],
                param_arr[self.standard_basis_idx['plx']],
                mtot
            ])
            xyz_params = self.standard_to_xyz(constrained_epoch, to_convert)

            # Update param_arr to hold xyz parameters
            param_arr[self.param_idx['x{}'.format(body_num)]] = xyz_params[0]
            param_arr[self.param_idx['y{}'.format(body_num)]] = xyz_params[1]
            param_arr[self.param_idx['z{}'.format(body_num)]] = xyz_params[2]
            param_arr[self.param_idx['xdot{}'.format(body_num)]] = xyz_params[3]
            param_arr[self.param_idx['ydot{}'.format(body_num)]] = xyz_params[4]
            param_arr[self.param_idx['zdot{}'.format(body_num)]] = xyz_params[5]
            param_arr[self.param_idx['plx']] = xyz_params[6]
            param_arr[self.param_idx['mtot']] = xyz_params[7]            

        return param_arr

    def standard_to_xyz(self, epoch, elems, tau_ref_epoch=58849, tolerance=1e-9, max_iter=100):
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
            elems = elems[:, np.newaxis]
        # Define variables
        sma = elems[0,:] # AU
        ecc = elems[1,:] # [0.0, 1.0]
        inc = elems[2,:] # rad [0, pi]
        aop = elems[3,:] # rad [0, 2 pi]
        pan = elems[4,:] # rad [0, 2 pi]
        tau = elems[5,:] # [0.0, 1.0]
        mtot = elems[7,:] # Msun

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
        pos_peri_z = np.zeros(len(pos_peri_x))

        pos = np.stack((pos_peri_x, pos_peri_y, pos_peri_z)).T
        pos_magnitude = np.linalg.norm(pos, axis=1)

        # Velocity vector in the perifocal system in km/s
        vel_peri_x = - ((np.sqrt(consts.G*(mtot*u.Msun)*(sma*u.AU))*np.sin(eanom) / (pos_magnitude*u.AU)).to(u.km / u.s)).value 
        vel_peri_y = ((h* np.cos(eanom) / (pos_magnitude*u.AU)).to(u.km / u.s)).value
        vel_peri_z = np.zeros(len(vel_peri_x))

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

        pos_xyz = np.zeros((len(sma), 3))
        vel_xyz = np.zeros((len(sma), 3))
        for k in range(len(sma)):
            pos_xyz[k,:] =  np.matmul(T[:,:,k], pos[k])
            vel_xyz[k,:] =  np.matmul(T[:,:,k], vel[k])

        result = np.stack([-pos_xyz[:,0], pos_xyz[:,1], pos_xyz[:,2], -vel_xyz[:,0], vel_xyz[:,1], vel_xyz[:,2], elems[6, :], mtot])

        if len(sma) == 1:
            result = result.T

        return np.squeeze(result)

    def verify_params(self):
        '''
        For now, additionally throws exceptions if data is supplied in sep/pa or if the best epoch for each
        body is one of the last two (which would inevtably mess up how the priors are imposed).
        '''
        super(XYZ, self).verify_params()

        # For now, raise error if data is in sep/pa
        seppa_locs = np.where(self.data_table['quant_type'] == 'seppa')
        if np.size(seppa_locs) != 0:
            raise Exception("For now, the XYZ basis requires data in RA and DEC offsets.")

        # For now, raise error if the best epoch for each body is one of the last two
        for i in range(self.num_secondary_bodies):
            body_num = i + 1
            best_epoch_loc = self.best_epoch_idx[i]
            body_indices = np.where(self.data_table['object'] == body_num)[0]
            max_index = np.amax(body_indices)

            if (max_index - best_epoch_loc < 2):
                raise Exception("For now, the epoch with the lowest sepparation error should not be one of the last two entries for body{}".format(body_num))


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
        float or np.array: corresponding t_p of the taus
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
        float or np.array: corresponding taus
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
        float or np.array: new taus
    """
    
    tp = tau_to_tp(old_tau, old_epoch, period)
    new_tau = tp_to_tau(tp, new_epoch, period)

    return new_tau

def tau_to_manom(date, sma, mtot, tau, tau_ref_epoch):
    """
    Gets the mean anomlay. Wrapper for kepler.tau_to_manom, kept here
    for backwards compatibility.
    
    Args:
        date (float or np.array): MJD
        sma (float): semi major axis (AU)
        mtot (float): total mass (M_sun)
        tau (float): epoch of periastron, in units of the orbital period
        tau_ref_epoch (float): reference epoch for tau
        
    Returns:
        float or np.array: mean anomaly on that date [0, 2pi)
    """

    return kepler.tau_to_manom(date, sma, mtot, tau, tau_ref_epoch)