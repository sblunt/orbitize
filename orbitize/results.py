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

    The `post` array is in the following order:

        semimajor axis 1, eccentricity 1, AOP 1, PAN 1, inclination 1, EPP 1,
        [semimajor axis 2, eccentricity 2, etc.],
        [total mass, parallax]

    where 1 corresponds to the first orbiting object, 2 corresponds
    to the second, etc. If stellar mass

    (written): Sarah Blunt, Henry Ngo, 2018
    """
    def __init__(self, sampler_name=None, post=None, lnlike=None):
        self.sampler_name = sampler_name
        self.post = post
        self.lnlike = lnlike

    def add_orbits(self, orbital_params, lnlikes):
        """
        Add accepted orbits and their likelihoods to the results

        Args:
            orbital_params (np.array): add sets of orbital params (could be multiple) to results
            lnlike (np.array): add corresponding lnlike values to results
        """
        # If no exisiting results then it is easy
        if self.post is None and self.lnlike is None:
            self.post = orbital_params
            self.lnlike = lnlikes
        # Otherwise, need to append properly
        else:
            self.post = np.vstack((self.post,orbital_params))
            self.lnlike = np.vstack((self.post,lnlikes))

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
