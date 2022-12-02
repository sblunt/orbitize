import orbitize.driver
import matplotlib.pyplot as plt

def comp_plt(results1, results2):
    """results1 is posteriors of first fit, results2 is second.
    """
    fig, axs = plt.subplots(3, 2)
    # fig.subplots_adjust(bottom = 0.1, top = 0.9, right = 0.15, left = 0.1) 

    axs[0][0].set_xlabel(r'$a ( AU )$')
    axs[0][0].plot(a_post)

    axs[1][0].set_xlabel(r'$i ( \deg )$')
    axs[1][0].plot()

    axs[2][0].set_xlabel(r'$\omega$')
    axs[2][0].plot()

    axs[0][1].set_xlabel(r'$e$')
    axs[0][1].plot()

    axs[1][1].set_xlabel(r'$\Omega (\deg)$')
    axs[1][1].plot()

    axs[2][1].set_xlabel(r'$T_0$')
    axs[2][1].plot()

    plt.show()