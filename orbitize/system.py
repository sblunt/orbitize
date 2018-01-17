
class System(object):
    """
    A gravitationally bound system

    Args:
        num_bodies (int): number of bodies in the system. Should be at least 2.
    """
    def __init__(self, num_bodies, system_mass, plx, mass_err=0, plx_err=0):
        pass

    def 


class CelestialDuo(object):
    """
    A 2-body gravitational interaction

    Args:
        data_table (astropy.table.Table): table of data on the 2-body

    """
    def __init__(self, data_table, orbital_param_type=None)
 
        pass

    def compute_orbit(self, orbital_parameters):
        """
        Comptue the orbit at the requested times

        Args:
            orbital_parameters (np.array): 1-D or 2-D array of orbital paramters. The last dimension has the dimensions of 1 set of orbital parameters. Can pass in multiple sets

        Returns:
            predictions (np.array): array of size (*, 6) where the first dimension only exists if orbital parameters is alos 2-D
        """
        pass

