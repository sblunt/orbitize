import orbitize
import matplotlib.pyplot as plt
import numpy as np
from orbitize import priors

#test gaussian prior draw samples
mu = 0.5
sigma = 0.01
num_samples = 1000
gauss = priors.GaussianPrior(mu, sigma)
gauss_samp = gauss.draw_samples(num_samples)
analytical = np.random.normal(mu, sigma, num_samples)
analytical = np.sort(analytical)
plt.hist(gauss_samp)
plt.hist(analytical, alpha = 0.5)
#x = np.linspace(0.,10, num_samples)
plt.show()

#test uniform prior draw samples
# minval = 10
# maxval = 15
# uniform = priors.UniformPrior(minval, maxval)
# uniform_samp = uniform.draw_samples(1000)
# plt.hist(uniform_samp)
# plt.show()