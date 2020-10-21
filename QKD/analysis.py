import astropy.units as u
import numpy as np
from uncertainties import unumpy, ufloat

from coincidence_count import count_coincidences

RATE_UNIT = 'Hz'

# n = count_coincidences('data/x0a0_y0b0.txt', plot=True)

names= np.array([[['HH', 'HV'], ['VH', 'VV']], [['AA', 'AD'], ['DA', 'DD']]])

def compute_rates():

    rates = unumpy.uarray(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))

    for basis in range(2):
        for pol_1 in range(2):
            for pol_2 in range(2):
                name = f'data/{names[basis, pol_1, pol_2]}_correct.txt'
                n, t = count_coincidences(name)
                rates[basis, pol_1, pol_2] = ufloat(
                    (n / t).to(RATE_UNIT).value, (np.sqrt(n) / t).to(RATE_UNIT).value)

    return(rates)


def QBER(basis, rates):
    data = rates[basis, :, :]
    total_counts = np.sum(data)
    counts = data[0, 1] + data[1, 0]
    return (counts / total_counts)

