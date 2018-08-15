class Results(object):
    """
    A class to store accepted orbital configurations from the sampler

    Args:
        post (np.array of float): MxN array of orbital parameters
            (posterior output from orbit-fitting process), where M
            is the number of varying orbital parameters in the fit, 
            and N is the number of orbits generated [None].
        mass_err (float [optional]): uncertainty on ``system_mass``, in M_sol
        plx_err (float [optional]): uncertainty on ``plx``, in mas


    The `post` array is in the following order:

        semimajor axis 1, eccentricity 1, AOP 1, PAN 1, inclination 1, EPP 1, 
        [semimajor axis 2, eccentricity 2, etc.],
        [total mass, parallax]

    where 1 corresponds to the first orbiting object, 2 corresponds
    to the second, etc. If stellar mass 

    """
    def __init__(self, post=None, mass_err=0, plx_err=0,):

        self.post = post
        self.mass_err = mass_err
        self.plx_err = plx_err

    def add_orbits(self, orbital_params):
        """
        Add accepted orbits to the results

        Args:
            orbital_params (np.array): add sets of orbital params (could be multiple) to results
        """
        pass

    def save_results(self, filename):
        """
        Save results to file

        Args:
            filename (string): filepath to save to
        """
        pass

    def plot_corner(self):
        pass

    def plot_orbit(self):
        pass

    