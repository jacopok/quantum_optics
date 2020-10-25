import astropy.units as u
import numpy as np
from uncertainties import unumpy, ufloat

from coincidence_count import count_coincidences

RATE_UNIT = 'Hz'

# n = count_coincidences('data/x0a0_y0b0.txt', plot=True)

def file_name(multi_index):
    x, y, a, b = multi_index
    return(f'data/x{x}a{a}_y{y}b{b}.txt')
    

def compute_rates():

    sigmas = np.zeros((2, 2, 2, 2))
    means = np.zeros((2, 2, 2, 2))

    for multi_index, _ in np.ndenumerate(means):
        mean, std = count_coincidences(file_name(multi_index), return_params=True)
        means[multi_index] = mean
        sigmas[multi_index] = std

    total_mean = np.average(means)
    total_std = np.average(sigmas)

    rates = unumpy.uarray(np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2, 2)))

    for multi_index, _ in np.ndenumerate(rates):
        n, t = count_coincidences(file_name(multi_index), total_mean, total_std)
        rates[multi_index] = ufloat(
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