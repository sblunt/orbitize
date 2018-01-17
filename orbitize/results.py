class Results(object):
    """
    Stores accepted orbital configurations from the sampler
    """
    def __init__(self):
        pass

    def add_orbits(self, orbital_parmas):
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

    