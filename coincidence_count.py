import numpy as np
import pandas as pd
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'''\usepackage{amsmath}
          \usepackage{physics}
          \usepackage{siunitx}
          ''')

RESOLUTION = 80.955 * u.ps * 2
THR = (-40, 80)
SIGMA_MULTIPLIER = 5


def read_file(name):
    """Returns a pandas dataframe from the comma-separated file at name"""

    return pd.read_csv(name,
                       sep=';',
                       header=None,
                       names=['ticks', 'channel'],
                       comment='#',
                       dtype=np.int)


def get_ticks(name):
    """Returns the arrays of ticks contained in the file at 'name'"""

    data = read_file(name)
    channels = set(data['channel'])

    try:
        channel_a, channel_b = channels
    except IndexError:
        print('More than two channels!')
        return (None)

    ticks_a = data[data['channel'] == channel_a]['ticks'].values // 2
    ticks_b = data[data['channel'] == channel_b]['ticks'].values // 2

    first_tick = min(ticks_a[0], ticks_b[0])

    return (ticks_a - first_tick, ticks_b - first_tick)


def get_timediffs(ticks_a, ticks_b, thr=THR):
    for tick in ticks_a:
        i = np.searchsorted(ticks_b, tick, side='left')
        try:
            if abs(tick - ticks_b[i - 1]) < abs(tick - ticks_b[i]):
                res = ticks_b[i - 1] - tick
            else:
                res = ticks_b[i] - tick
        except IndexError:
            res = ticks_b[i] - tick

        if thr[0] <= res <= thr[1]:
            yield res


def timediffs_histo(ticks_a, ticks_b, thr=THR):

    time_diffs = get_timediffs(ticks_a, ticks_b, thr)
    bins = np.arange(*thr)
    vals = np.zeros_like(bins)

    for x in time_diffs:
        vals[bins == x] += 1
    return (bins, vals)


def get_statistics(bins, vals):

    def model(x, mean, std, const): return const * \
        norm(loc=mean, scale=std).pdf(x)

    popt, _, _, _, _ = curve_fit(model, bins, vals, p0=[20, 10, 1000])

    # mean = np.average(bins, weights=vals)
    # var = np.average((bins - mean)**2, weights=vals)
    return (popt[0], popt[1])


def plot_timediffs(bins, vals,
                   minimal_bins, minimal_vals):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = iter(prop_cycle.by_key()['color'])

    c1 = next(colors)
    plt.bar(bins, vals, alpha=.5, color=c1)
    plt.bar(minimal_bins, minimal_vals, label="Time differences",
            alpha=.2, color=c1)
    plt.xlabel('Time difference [integer multiples of 160ps]')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()


def select_coincidences(ticks, sigma_multiplier, plot):

    bins, vals = timediffs_histo(*ticks)
    mean, std = get_statistics(bins, vals)

    minimal_thr = (int(round(mean - std * sigma_multiplier)),
                   int(round(mean + std*sigma_multiplier)))

    minimal_bins, minimal_vals = timediffs_histo(*ticks, thr=minimal_thr)

    if plot:
        plot_timediffs(bins, vals, minimal_bins, minimal_vals)
    
    return (minimal_bins, minimal_vals)
    
def count_coincidences(name, sigma_multiplier=SIGMA_MULTIPLIER, plot=False):
    
    ticks = get_ticks(name)
    minimal_bins, minimal_vals = select_coincidences(ticks, sigma_multiplier, plot)
    
    return(np.sum(minimal_vals))