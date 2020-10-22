import astropy.units as u
import numpy as np
from uncertainties import unumpy, ufloat

from coincidence_count import count_coincidences

RATE_UNIT = 'Hz'

# n = count_coincidences('data/x0a0_y0b0.txt', plot=True)

names = np.array([[['HH', 'HV'], ['VH', 'VV']], [['AA', 'AD'], ['DA', 'DD']]])


def compute_rates():
    
    sigmas = np.zeros((2, 2, 2))
    means = np.zeros((2, 2, 2))
    
    for basis in range(2):
        for pol_1 in range(2):
            for pol_2 in range(2):
                name = f'data/{names[basis, pol_1, pol_2]}_correct.txt'
                mean, std = count_coincidences(name, return_params=True)
                means[basis, pol_1, pol_2] = mean
                sigmas[basis, pol_1, pol_2] = std

    total_mean = np.average(means)
    total_std = np.average(sigmas)
    
    rates = unumpy.uarray(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))

    for basis in range(2):
        for pol_1 in range(2):
            for pol_2 in range(2):
                name = f'data/{names[basis, pol_1, pol_2]}_correct.txt'
                count, time = count_coincidences(name, total_mean, total_std)
                rates[basis, pol_1, pol_2] = ufloat(
                    (count / time).to(RATE_UNIT).value,
                    (np.sqrt(count) / time).to(RATE_UNIT).value)

    return(rates)


def QBER(basis, rates):
    data = rates[basis, :, :]
    total_counts = np.sum(data)
    counts = data[0, 1] + data[1, 0]
    return (counts / total_counts)

if __name__ == "__main__":
    
    rates = compute_rates()
    print(f'QBER for the H-V basis: {QBER(0, rates)}')
    print(f'QBER for the A-D basis: {QBER(1, rates)}')