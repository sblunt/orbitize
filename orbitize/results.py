import matplotlib.pyplot as plt
import corner

class Results(object):
    """
    A class to store accepted orbital configurations from the sampler

    Args:
        sampler_name (string): name of sampler class that generated these results [None].
        post (np.array of float): MxN array of orbital parameters
            (posterior output from orbit-fitting process), where M
            is the number of varying orbital parameters in the fit,
            and N is the number of orbits generated [None].
        lnlike (np.array of float): N array of ln-likelihoods corresponding to
            the orbits described in post [None].
        mass_err (float [optional]): uncertainty on ``system_mass``, in M_sol
        plx_err (float [optional]): uncertainty on ``plx``, in mas

    The `post` array is in the following order:

        semimajor axis 1, eccentricity 1, argument of periastron 1,
        position angle of nodes 1, inclination 1, epoch of periastron passage 1,
        [semimajor axis 2, eccentricity 2, etc.],
        [total mass, parallax]

    where 1 corresponds to the first orbiting object, 2 corresponds
    to the second, etc. If stellar mass

    (written): Sarah Blunt, Henry Ngo, 2018
    """
    def __init__(self, sampler_name=None, post=None, lnlike=None, mass_err=0, plx_err=0,):
        self.sampler_name = sampler_name
        self.post = post
        self.lnlike = lnlike
        self.mass_err = mass_err
        self.plx_err = plx_err

    def add_orbits(self, orbital_params, lnlikes):
        """
        Add accepted orbits and their likelihoods to the results

        Args:
            orbital_params (np.array): add sets of orbital params (could be multiple) to results
            lnlike (np.array): add corresponding lnlike values to results

        (written): Henry Ngo, 2018
        """
        # If no exisiting results then it is easy
        if self.post is None and self.lnlike is None:
            self.post = orbital_params
            self.lnlike = lnlikes
        # Otherwise, need to append properly
        else:
            pass # TODO

    def save_results(self, filename):
        """
        Save results to file

        Args:
            filename (string): filepath to save to
        """
        pass

    def plot_corner(self, param_list=[]):
        """
        Make a corner plot of posterior on orbit fit from any sampler

        Args:
            param_list (list of strings): each entry is a name of a parameter to include
                valid strings:
                sma1: semimajor axis
                ecc1: eccentricity
                inc1: inclination
                aop1: argument of periastron
                pan1: position angle of nodes
                epp1: epoch of periastron passage
                [repeat for 2, 3, 4, etc if multiple objects]
                mtot: total mass
                plx:  parallax
                e.g. Use param_list = ['sma1,ecc1,inc1,sma2,ecc2,inc2'] to only
                     plot posteriors for semimajor axis, eccentricity and inclination
                     of the first two companions

        Return:
            matplotlib.pyplot Figure object of the corner plot

        (written): Henry Ngo, 2018
        """
        if len(param_list)>0: # user chose to plot specific parameters only
            num_orb_param = self.post.shape[0] # number of orbital parameters (+ mass, parallax)
            num_objects = np.trunc(num_orb_param / 6).astype(np.int)
            

        semimajor axis 1, eccentricity 1, argument of periastron 1,
        position angle of nodes 1, inclination 1, epoch of periastron passage 1,
        [semimajor axis 2, eccentricity 2, etc.],
        [total mass, parallax]

        figure = corner.corner(self.post)
        return figure

    def plot_orbit(self, n_orbits=100):
        """
        Make plots of selected orbits

        Args:
            n_orbits (int): number of orbits to plot

        Return:
            matplotlib.pyplot Figure object of the orbit plot

        (written): Henry Ngo, 2018
        """
        pass
