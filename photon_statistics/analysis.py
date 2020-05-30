from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import pandas as pd
import numba
from matplotlib import cm
from scipy.stats import poisson
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'''\usepackage{amsmath}
          \usepackage{physics}
          \usepackage{siunitx}
          ''')


THERMAL_PATH = 'data/24, Jan, 2020 - Thermal/'
COHERENT_PATH = 'data/24, Jan, 2020 - Coherent/'
RESOLUTION = 81 * u.picosecond
MAX_TICKS = int(1e4)

WINDOWS = np.logspace(-2, 3.5, num=100) * u.us

THERMAL_NAMES = [THERMAL_PATH + 'Part_' + str(i) + '.txt' for i in range(10)]
COHERENT_NAMES = [COHERENT_PATH + 'Part_' + str(i) + '.txt' for i in range(10)]

COLORS = {
    'mode': 'purple',
    'mean': 'lime',
    'variance': 'red',
    'theoretical variance': 'red',
    'skewness': 'blue',
    'theoretical skewness': 'blue',
    'kurtosis': 'green',
    'theoretical kurtosis': 'green'
}


def read_file(name):
    """Returns a pandas dataframe from the comma-separated file at name"""

    return pd.read_csv(name,
                       sep=',',
                       header=None,
                       names=['ticks', 'channel'],
                       dtype=np.int)


def get_ticks(name):
    """Returns the array of times contained in the file at name"""
    data = read_file(name)
    return data['ticks'].values


def get_ticks_names(names):
    """Returns a list of arrays of times from a list of names.
    All of these are normalized so that the first time is zero.
    """

    all_ticks = [get_ticks(name) for name in names]
    return [x - x[0] for x in all_ticks]


ALL_TICKS_ARRAYS = {
    'thermal': get_ticks_names(THERMAL_NAMES),
    'coherent': get_ticks_names(COHERENT_NAMES)
}


def sum_arrays(arrays):
    """Given a list of arrays, returns an array as long as the
    longest of these, containing the sum of all the arrays
    at corresponding indices.
    """

    max_len = max(len(a) for a in arrays)
    result = np.zeros(max_len)
    for a in arrays:
        result[:len(a)] += a
    return result


@numba.jit(nopython=True)
def _base_get_n_in_window(ticks, adim_window):
    """Helper function - returns the number of events observed in a specific window size
    for a single array.

    Arguments:
    ticks -- an array of time measurement.
    adim_window -- window size in the same units as the time measurements.

    Returns:
    counts -- an array whose i-th element is the number of windows of size
        adim_window in which i events were found.
    """
    tmax = ticks[-1]
    number_of_windows = int(tmax / adim_window)
    nums = np.zeros(number_of_windows)

    indices = np.searchsorted(ticks,
                              np.arange(number_of_windows + 1) * adim_window)
    nums = np.ediff1d(indices)
    counts = np.zeros(max(nums) + 1)
    for num in nums:
        counts[num] += 1
    return counts


def get_n_in_window_from_ticks(ticks, window, resolution):
    """Returns the number of events observed in a specific window size
    for a single array.
    May use a lot of memory for small window sizes (less than 10ns).

    Arguments:
    ticks -- an array of time measurements.
    window -- window size - must be an astropy quantity which can be converted to
        a time measurement.

    Keyword arguments:
    resolution -- conversion factor between the integer ticks and physical times,
        one tick should correspond to a time equal to this constant.
        Must be an astropy quantity which can be converted to a time measurement.

    Returns:
    counts -- an array whose i-th element is the number of windows of size
        adim_window in which i events were found.
    """

    len_ticks = len(ticks)
    total_ticks = MAX_TICKS

    number_subdivisions = len_ticks // total_ticks
    adim_window = (window / resolution).to(u.dimensionless_unscaled).value

    all_nums = []
    for subdivision in range(number_subdivisions):
        current_ticks = ticks[subdivision * total_ticks:(subdivision + 1) *
                              total_ticks]
        nums = _base_get_n_in_window(current_ticks - current_ticks[0],
                                     adim_window)
        all_nums.append(nums)

    return sum_arrays(all_nums)


def get_n_in_window_from_all_ticks(all_ticks, window, resolution=RESOLUTION):
    """Returns the number of events observed in a specific window size
    for a list of arrays.
    May use a lot of memory for small window sizes (less than 10ns).
    Uses multiprocessing with the number of cpus available minus one.

    Arguments:
    all_ticks -- a list of arrays of time measurements.
    window -- window size - must be an astropy quantity which can be converted to
        a time measurement.

    Keyword arguments:
    resolution -- conversion factor between the integer ticks and physical times,
        one tick should correspond to a time equal to this constant.
        Must be an astropy quantity which can be converted to a time measurement.
        Defaults to the global variable RESOLUTION.

    Returns:
    counts -- an array whose i-th element is the number of windows of size
        adim_window in which i events were found, across all of the arrays of ticks.
    """

    print(f'Analyzing window size {window}')
    pool = Pool(cpu_count() - 1)
    func = partial(get_n_in_window_from_ticks,
                   window=window,
                   resolution=resolution)
    all_ns = pool.map(func, all_ticks)
    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks

    return sum_arrays(all_ns)


def get_photon_counts(window, ticks_dict):
    """Given a dictionary of lists of arrays of ticks and a window size,
    returns a dictionary of event counts, computed with
    `get_n_in_window_from_all_ticks`.
    """

    photon_counts = {}
    for name, ticks_arrays in ticks_dict.items():
        photon_counts[name] = get_n_in_window_from_all_ticks(
            ticks_arrays, window)
    return photon_counts


def thermal(counts, nbar):
    """Theoretical thermal distribution,
    with mean nbar.
    """
    return (nbar / (1 + nbar))**counts / (1 + nbar)


def coherent(counts, nbar):
    """Theoretical coherent distribution,
    with mean nbar
    for n counts.
    """

    return poisson.pmf(counts, nbar)


THEORETICAL_DISTRIBUTIONS = {
    'thermal': thermal,
    'coherent': coherent,
}


def moment(bins, values, order):
    """Unnormalized order-th moment for the probability mass function given by
    `bins` and `values`.
    """

    mean = np.average(bins, weights=values)

    if order == 1:
        return mean

    return np.sum((bins - mean)**order * values / np.sum(values))


def analyze(dist, bins, nbar):
    """Computes the theoretical distribution `dist` with mean nbar
    over the `bins` given, and returns a dictionary with a description
    of the moments of the theoretical distribution.
    """

    values = dist(bins, nbar)
    th_desc = {'variance': moment(bins, values, 2)}
    th_desc['skewness'] = moment(bins, values, 3) / moment(bins, values,
                                                           2)**(3 / 2)
    th_desc['kurtosis'] = moment(bins, values, 4) / moment(bins, values,
                                                           2)**(4 / 2)
    return th_desc


def describe(dist, dist_type, bins=None):
    """Given a distribution `dist` given as an order array of values,
    gives a description of the distribution, comparing it to the theoretical
    `dist_type`.
    """

    description = {}
    if bins is None:
        bins = range(len(dist))

    description['mode'] = np.argmax(dist)
    description['mean'] = moment(bins, dist, 1)

    th_desc = analyze(THEORETICAL_DISTRIBUTIONS[dist_type], bins,
                      description['mean'])

    description['variance'] = moment(bins, dist, 2)
    description['theoretical variance'] = th_desc['variance']
    description['skewness'] = (moment(bins, dist, 3) /
                               moment(bins, dist, 2)**(3 / 2))
    description['theoretical skewness'] = th_desc['skewness']
    description['kurtosis'] = (moment(bins, dist, 4) /
                               moment(bins, dist, 2)**(4 / 2))
    description['theoretical kurtosis'] = th_desc['kurtosis']

    return description


def plot_descriptions(windows, descriptions, colors=None):
    """Given a dictionary of lists of descriptions, 
    obtained with window sizes given in the list `windows`,
    plot them.

    Arguments:
    `windows` -- an array of u.Quantity, the window sizes
    at which the descriptions were computed.

    `descriptions` -- a dictionary, indexed by the physical type of distribution ('thermal', 'coherent'),
    of lists, with varying window size,
    of descriptions: dictionaries, indexed by the descriptor ('mean', 'variance' and so on)
    of the values of the descriptors.

    Keyword arguments:
    colors -- a dictionary, with indices corresponding to the names of the descriptors,
    of strings representing the colors with which the descriptors should be plotted.
    """

    _, axs = plt.subplots(1, 2, sharey=True)

    if colors is None:
        colors = COLORS

    for i, (name, description) in enumerate(descriptions.items()):
        for characteristic in description[0]:

            linestyle = '--' if 'theoretical' in characteristic else '-'

            axs[i].loglog(windows, [y[characteristic] for y in description],
                          label=characteristic,
                          ls=linestyle,
                          c=colors[characteristic])

        axs[i].set_title(name)
        axs[i].legend()
        axs[i].set_xlabel(f'window size [{windows.unit}]')
        axs[i].grid('on', which='major')
        # axs[i].tick_params(which='both')
        # axs[i].tick_params(axis='y', which='minor', bottom=False)
    # axs[0].yaxis.set_minor_locator(ticker.LogLocator())

        axs[i].minorticks_on()
    plt.tight_layout()
    plt.savefig('descriptions.pdf', format='pdf')
    plt.show(block=False)


def get_descriptions(windows=None, ticks_arrays=None):
    """Returns a dictionary of lists of descriptions.

    Arguments:
    `windows` -- an array of u.Quantity, the window sizes
    at which the descriptions are to be computed.
    Defaults to WINDOWS.

    `ticks_arrays` --  a dictionary, indexed by the distribution type,
    of lists of ticks coming from different files.
    Defaults to ALL_TICKS_ARRAYS.
    """

    if ticks_arrays is None:
        ticks_arrays = ALL_TICKS_ARRAYS
    if windows is None:
        windows = WINDOWS

    descriptions = defaultdict(list)

    for window in windows:
        photon_counts = get_photon_counts(window, ticks_arrays)

        for n, dist in photon_counts.items():
            description = describe(dist, n)
            descriptions[n].append(description)

    return descriptions


def get_rate(descriptions, windows, unit=u.kHz):
    """ Given the `descriptions` (a dictionary of lists of dictionaries of characteristics)
    and `windows`, the list of window sizes, returns a dictionary of estimates of the rates of events,
    as well as the ratio of the 'thermal' to the 'coherent' rates.

    Arguments:
    `descriptions` -- a dictionary, indexed by the physical type of distribution ('thermal', 'coherent'),
    of lists, with varying window size,
    of descriptions: dictionaries, indexed by the descriptor ('mean', 'variance' and so on)
    of the values of the descriptors.
    """

    rates = {}
    for n, description in descriptions.items():
        distribution_rates = []
        for window, desc in zip(windows.value, description):
            mean = desc['mean']
            rate = mean / window
            distribution_rates.append(rate)
        rates[n] = np.average(distribution_rates) / windows.unit
        if unit is not None:
            rates[n] = rates[n].to(unit)
    rates['ratio'] = (rates['thermal'] / rates['coherent']).to(u.percent)

    return rates


if __name__ == '__main__':
    w = 100 * u.us
    photon_counts = get_photon_counts(w, ALL_TICKS_ARRAYS)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = iter(prop_cycle.by_key()['color'])
    plt.close()
    for this_name, distribution in photon_counts.items():
        n = range(len(distribution))
        c = next(colors)
        m = np.average(n, weights=distribution)
        t = np.sum(distribution)
        plt.bar(n, distribution/t, label=this_name, alpha=.5, color=c)
        plt.plot(n, THEORETICAL_DISTRIBUTIONS[this_name](n, m), color=c)
        a, b = plt.xlim()
        plt.xlim((-1, b/1.5))
        plt.ylabel('Probability')
        plt.xlabel('Number of photons')
        plt.title(f'Window: {w}')
    plt.legend()
    plt.savefig(f'../lab_report/figures/{w.value}{w.unit}.pdf', format='pdf', dpi=250)
