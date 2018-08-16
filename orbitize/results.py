import numpy as np
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

    The `post` array is in the following order:

        semimajor axis 1, eccentricity 1, argument of periastron 1,
        position angle of nodes 1, inclination 1, epoch of periastron passage 1,
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

        (written): Henry Ngo, 2018
        """
        # If no exisiting results then it is easy
        if self.post is None and self.lnlike is None:
            self.post = orbital_params
            self.lnlike = lnlikes
        # Otherwise, need to append properly
        else:
            self.post = np.vstack((self.post,orbital_params))
            self.lnlike = np.vstack((self.lnlike,lnlikes))

    def save_results(self, filename):
        """
        Save results to file

        Args:
            filename (string): filepath to save to
        """
        pass

    def plot_corner(self, param_list=[], **corner_kwargs):
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
            **corner_kwargs: any remaining keyword args are sent to corner.corner
                             See: https://corner.readthedocs.io/

        Return:
            matplotlib.pyplot Figure object of the corner plot

        (written): Henry Ngo, 2018
        """
        if len(param_list)>0: # user chose to plot specific parameters only
            num_orb_param = self.post.shape[0] # number of orbital parameters (+ mass, parallax)
            num_objects,remainder = np.divmod(num_orb_param,6)
            have_mtot_and_plx = remainder == 2
            # Define a dictionary to look up index of certain parameters
            dict_of_indices = {
                'sma' = 0
                'ecc' = 1
                'inc' = 2 ## CHECK THAT ORDER IS CORRECT
                'aop' = 3
                'pan' = 4
                'epp' = 5
            }
            param_indices = []
            for param in param_list:
                if param=='mtot':
                    if have_mtot_and_plx:
                        param_indices.append(num_orb_param-2) # the 2nd last index
                elif param=='plx':
                    if have_mtot_and_plx:
                        param_indices.append(num_orb_param-2) # the last index
                elif len(param)==4: # to prevent invalid, short param names breaking
                    if param[0:3] in dict_of_indices:
                        object_id = np.int(param[3])
                        index = dict_of_indices[param[0:3]] + 6*(object_id-1)
                        param_indices.append(index)
                else:
                    pass # skip unrecognized parameter
            samples = self.post[param_indices,:] # Keep only chains for selected parameters

        else:
            samples = self.post

        figure = corner.corner(samples, **corner_kwargs)
        return figure

    def plot_orbits(self, n_orbits=100):
        """
        Make plots of selected orbits

        Args:
            n_orbits (int): number of orbits to plot

        Return:
            matplotlib.pyplot Figure object of the orbit plot

        (written): Henry Ngo, Sarah Blunt, 2018
        """
        # TODO: modify this line to work with actual orbitize outputs
        sma, ecc, tp, w, pan, inc, prlx, mtot, mplanet = orbitize_outputs
        ##

        choose = np.random.randint(0, high=len(sma), size=num_orbits2plot)

        raoff = np.zeros((num_orbits2plot, num_epochs))
        deoff = np.zeros((num_orbits2plot, num_epochs))
        epochs = np.zeros((num_orbits2plot, num_epochs))

        _ = plt.figure()
        colormap = cm.inferno

        # TODO: could probably remove this for loop, haven't checked calc_orbit in a while
        for i in np.arange(num_orbits2plot):
            epochs[i,:] = np.linspace(start_date, float(start_date+per[choose[i]]), num_epochs)
            raoff0, deoff0, _ = calc_orbit(
                epochs[i,:], sma[choose[i]], ecc[choose[i]], tp[choose[i]], w[choose[i]], pan[choose[i]],
                inc[choose[i]], prlx[choose[i]], mtot[choose[i]], mass=mplanet[choose[i]]
            )
            raoff[i,:] = raoff0
            deoff[i,:] = deoff0

        latest_time = np.max(epochs)
        for i in np.arange(num_orbits2plot):
            for j in np.arange(num_epochs-2):

                plt.plot(raoff[i, j:j+2], deoff[i, j:j+2], color=cm.inferno(epochs[i,j]/latest_time))

            plt.plot([raoff[i,-1], raoff[i,0]], [deoff[i,-1], deoff[i,0]], color=colormap(epochs[i,-1]/latest_time))

        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('$\Delta$RA (mas)')
        ax.set_ylabel('$\Delta$Dec (mas)')
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=6)

        # TODO: add color bar
