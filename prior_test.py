import orbitize
import matplotlib.pyplot as plt
import numpy as np
from orbitize import priors

# test gaussian prior draw samples
# mu = 0.5
# sigma = 0.01
# num_samples = 1000
# gauss = priors.GaussianPrior(mu, sigma)
# gauss_samp = gauss.draw_samples(num_samples)
# analytical = np.random.normal(mu, sigma, num_samples)
# analytical = np.sort(analytical)
# plt.hist(gauss_samp)
# plt.hist(analytical, alpha = 0.5)
# plt.savefig("/home/tmckenna/orbitize/gauss_test.png")

# test uniform prior draw samples
# minval = 10
# maxval = 15
# uniform = priors.UniformPrior(minval, maxval)
# uniform_samp = uniform.draw_samples(1000)
# plt.hist(uniform_samp)
# plt.savefig("/home/tmckenna/orbitize/uniform_test.png")

# test sin prior draw samples

sin = priors.SinPrior()
sin_samp = sin.draw_samples(1000)
x = np.linspace(0, np.pi)
y = np.sin(x)
plt.plot(x,y)
plt.hist(sin_samp, density = True)
plt.savefig("/home/tmckenna/orbitize/sin_test.png")

# test log uniform prior draw samples
# minval = 0.1
# maxval = 100
# log = priors.LogUniformPrior(minval, maxval)
# log_samp = log.draw_samples(10000)
# plt.hist(log_samp, bins = np.logspace(-1, 2, 100))
# plt.xscale("log")
# plt.savefig("/home/tmckenna/orbitize/log_test.png")