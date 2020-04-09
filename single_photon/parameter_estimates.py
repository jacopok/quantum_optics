import numpy as np
import astropy.units as u
from bayesian_hyp_testing import uniform_dist_log, uniform_dist
from simulation import detections

RESOLUTION = 80.955 * u.picosecond

N_G = 1_554_341
N_G1 = 13_839  # reflected
N_G2 = 14_821  # transmitted
N_G12 = 2

meas = detections(N_G1, N_G2, N_G12, N_G)
ratio = N_G2 / N_G1

rate, rate_pdf = uniform_dist(np.exp(-4.4), np.exp(-3.95), num=50)
e_rate, e_rate_pdf = uniform_dist_log(-15, -4, num=50)

param_prior = np.outer(rate_pdf, e_rate_pdf)
