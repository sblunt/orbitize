"""
Module to test plotting functions not tested by other tests/results testing.
"""
import os
import orbitize
import orbitize.driver
import orbitize.plot as plot

def test_plot_residuals(debug=False):
    """
    Tests the residual plotting code on a 2-planet orbit fit
    """
    # run a very short 2 planet orbit fit
    input_file = os.path.join(orbitize.DATADIR, "GJ504.csv")
    my_driver = orbitize.driver.Driver(input_file, 
                                       "OFTI", 
                                       1, 
                                       1.22, 
                                       56.95, 
                                       mass_err=0.08, 
                                       plx_err=0.26,
                                       system_kwargs={"restrict_angle_ranges": True}
                                       )
    my_sampler = my_driver.sampler
    my_sampler.run_sampler(101)
    my_results = my_sampler.results

    # plot planet 1
    fig1 = plot.plot_residuals(my_results)

    # plot with more samples, mod 180
    fig2 = plot.plot_residuals(my_results, object_to_plot=1, mod180=True, num_orbits_to_plot=150)

    if debug:
        import matplotlib.pylab as plt
        plt.show()

if __name__ == "__main__":
    test_plot_residuals(debug=True)

