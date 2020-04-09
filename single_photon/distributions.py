import numpy as np

R1 = R2 = 1e2

OBS_TIME = 1e3

SIZE = int(1e4)

# DELTA_T=1e-4
delta_ts = np.logspace(-5, 1)
averages = np.zeros_like(delta_ts)

for i, delta_t in enumerate(delta_ts):
    coincidence_rate = (1 - np.exp(-R1 * delta_t)) * (1 -
                                                      np.exp(-R2 * delta_t))

    N1 = np.random.poisson(lam=OBS_TIME * R1, size=SIZE)
    N2 = np.random.poisson(lam=OBS_TIME * R2, size=SIZE)

    N12 = np.random.binomial(n=int(OBS_TIME / delta_t),
                             p=coincidence_rate,
                             size=SIZE)

    g = N12 * SIZE / (N1 * N2)

    averages[i] = np.average(g)
