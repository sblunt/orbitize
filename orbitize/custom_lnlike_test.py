import custom_hd159062_lnlike as custom_lnlike
import numpy as np

def p2sma(P,m0,m1):
    return ((P/365.25)**2*(m0 + m1))**(1/3)

sma = p2sma(238*365.25,0.80,0.65)
ecc = 0.44
inc = 53.0*np.pi/180
argp = (-26+180)*np.pi/180
lan = 138.0*np.pi/180
tau = 0.0
plx = 46.12
m1 = 0.65
m0 = 0.80

wd_params = [sma,ecc,inc,argp,lan,tau,plx,m1,m0]

print('Custom Likelihood:',custom_lnlike.custom_chi2_loglike(wd_params))
