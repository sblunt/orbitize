import numpy as np
import astropy.units as u
import astropy.constants as consts
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

        semimajor axis 1, eccentricity 1, inclination 1,
        argument of periastron 1, position angle of nodes 1,
        epoch of periastron passage 1,
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
            self.lnlike = np.append(self.lnlike,lnlikes)

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
                             Note: default axis labels used unless overwritten by user input

        Return:
            matplotlib.pyplot Figure object of the corner plot

        (written): Henry Ngo, 2018
        """
        # Define a dictionary to look up index of certain parameters
        dict_of_indices = {
            'sma': 0,
            'ecc': 1,
            'inc': 2,
            'aop': 3,
            'pan': 4,
            'epp': 5
        }
        # Define array of default axis labels (overwritten if user specifies list)
        default_labels = [
            'a [au]',
            'ecc',
            'inc [deg]',
            '$\omega$ [deg]',
            '$\Omega$ [deg]',
            '$\\tau$',
            '$M_T$ [Msol]',
            '$\pi$ [mas]'
        ]
        if len(param_list)>0: # user chose to plot specific parameters only
            num_orb_param = self.post.shape[1] # number of orbital parameters (+ mass, parallax)
            num_objects,remainder = np.divmod(num_orb_param,6)
            have_mtot_and_plx = remainder == 2
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

            samples = self.post[:,param_indices] # Keep only chains for selected parameters
            if 'labels' not in corner_kwargs: # Use default labels if user didn't already supply them
                # Necessary to index a list with another list
                reduced_labels_list = [default_labels[i] for i in param_indices]
                corner_kwargs['labels'] = reduced_labels_list
        else:
            samples = self.post
            if 'labels' not in corner_kwargs: # Use default labels if user didn't already supply them
                corner_kwargs['labels'] = default_labels

        figure = corner.corner(samples, **corner_kwargs)
        return figure

    def plot_orbits(self, parallax, total_mass, object_mass, object_to_plot=1, start_date=2000, num_orbits_to_plot=100, num_epochs_to_plot=100):
        """
        Plots one orbital period for a select number of fitted orbits for a given object

        Args:
            parallax (float): parallax in mas, however, if plx_err was passed
                to system, then this is ignored and the posterior samples for
                plx will be used instead
            total_mass (float): total mass of system in solar masses, however,
                if mass_err was passed to system, then this is ignored and the
                posterior samples for mtot will be used instead
            object_mass (float): mass of the object, in solar masses
            object_to_plot (int): which object to plot [1]
            start_date (float): year in which to start plotting orbits
            num_orbits_to_plot (int): number of orbits to plot [100]
            num_epochs_to_plot (int): number of points to plot per orbit [100]

        Return:
            matplotlib.pyplot Figure object of the orbit plot if input valid, None otherwise

        (written): Henry Ngo, Sarah Blunt, 2018
        """
        # Split the 2-D post array into series of 1-D arrays for each orbital parameter
        num_objects, remainder = np.divmod(self.post.shape[0],6)
        if object_to_plot > num_objects:
            return None
        first_index = 0 + 6*(object_to_plot-1)
        sma = self.post[first_index+0,:]
        ecc = self.post[first_index+1,:]
        inc = self.post[first_index+2,:]
        aop = self.post[first_index+3,:]
        pan = self.post[first_index+4,:]
        epp = self.post[first_index+5,:]
        # Then, get the other parameters
        if remainder == 2: # have samples for parallax and mtot
            mtot = self.post[-2,:]
            plx = self.post[-1,:]
        else: # otherwise make arrays out of user provided value
            mtot = np.ones(len(sma))*total_mass
            plx = np.ones(len(sma))*parallax
        mplanet = np.ones(len(sma))*object_mass

        # Select random indices for plotted orbit
        if num_orbits_to_plot > len(sma):
            num_orbits_to_plot = len(sma)
        choose = np.random.randint(0, high=len(sma), size=num_orbits_to_plot)

        raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
        epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

        # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
        # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
        for i in np.arange(num_orbits_to_plot):
            orb_ind = choose[i]
            # Compute period (from Kepler's third law)
            period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
            period = period.to(u.year).value
            # Create an epochs array to plot num_epochs_to_plot points over one orbital period
            epochs[i,:] = np.linspace(start_date, float(start_date+period), num_epochs_to_plot)
            # Calculate ra/dec offsets for all epochs of this orbit
            raoff0, deoff0, _ = calc_orbit(
                epochs[i,:], sma[orb_ind], ecc[orb_ind], epp[orb_ind], aop[orb_ind], pan[orb_ind],
                inc[orb_ind], plx[orb_ind], mtot[orb_ind], mass=mplanet[orb_ind]
            )
            raoff[i,:] = raoff0
            deoff[i,:] = deoff0

        latest_time = np.max(epochs)

        orbit_figure = plt.figure()
        colormap = cm.inferno
        # Plot each orbit
        for i in np.arange(num_orbits_to_plot):
            for j in np.arange(num_epochs-2):

                plt.plot(raoff[i, j:j+2], deoff[i, j:j+2], color=cm.inferno(epochs[i,j]/latest_time))

            plt.plot([raoff[i,-1], raoff[i,0]], [deoff[i,-1], deoff[i,0]], color=colormap(epochs[i,-1]/latest_time))

        # Modify the axes
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('$\Delta$RA (mas)')
        ax.set_ylabel('$\Delta$Dec (mas)')
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=6)

        # Add colorbar
        plt.colorbar(cax=ax)

        return orbit_figure
