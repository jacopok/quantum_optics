import astropy.units as u
import numpy as np
from uncertainties import unumpy, ufloat

from coincidence_count import count_coincidences

RATE_UNIT = 'Hz'

# n = count_coincidences('data/x0a0_y0b0.txt', plot=True)


def compute_rates():

    sigmas = np.zeros((2, 2, 2, 2))
    means = np.zeros((2, 2, 2, 2))
    
    for x in range(2):
        for y in range(2):
            for a in range(2):
                for b in range(2):
                    name = f'data/x{x}a{a}_y{y}b{b}.txt'
                    mean, std = count_coincidences(name, return_params=True)
                    means[x, y , a, b] = mean
                    sigmas[x, y , a, b] = std

    total_mean = np.average(means)
    total_std = np.average(sigmas)

    rates = unumpy.uarray(np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2, 2)))

    for x in range(2):
        for y in range(2):
            for a in range(2):
                for b in range(2):
                    name = f'data/x{x}a{a}_y{y}b{b}.txt'
                    n, t = count_coincidences(name, total_mean, total_std)
                    rates[x, y, a, b] = ufloat(
                        (n / t).to(RATE_UNIT).value, (np.sqrt(n) / t).to(RATE_UNIT).value)

    return(rates)


def expected_value(x, y, rates):
    data = rates[x, y, :, :]
    total_counts = np.sum(data)
    counts = data[0, 0] + data[1, 1] - data[0, 1] - data[1, 0]
    return (counts / total_counts)


def CHSH(rates):
    return (
        expected_value(0, 0, rates) +
        expected_value(0, 1, rates) +
        expected_value(1, 0, rates) -
        expected_value(1, 1, rates)
    )

if __name__ == "__main__":
    
    computed_rates = compute_rates()
    
    print(f'CHSH = {CHSH(computed_rates)}')