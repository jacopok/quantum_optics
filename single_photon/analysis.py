import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

FILENAME = 'data/TimeTags.txt'
RESOLUTION = 80.955 * u.ps * 2
THR = (-40, 80)

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'''\usepackage{amsmath}
          \usepackage{physics}
          \usepackage{siunitx}
          ''')

def read_file(name):
    """Returns a pandas dataframe from the comma-separated file at name"""

    return pd.read_csv(name,
                       sep=';',
                       header=None,
                       names=['ticks', 'channel'],
                       comment='#',
                       dtype=np.int)


def get_ticks(name=FILENAME):
    """Returns the arrays of times contained in the file at name"""

    data = read_file(name)

    ticks_t = data[data['channel'] == 2]['ticks'].values // 2
    ticks_r = data[data['channel'] == 3]['ticks'].values // 2
    ticks_g = data[data['channel'] == 4]['ticks'].values // 2

    first_tick = min(ticks_t[0], ticks_r[0], ticks_g[0])

    return (ticks_t - first_tick, ticks_r - first_tick, ticks_g - first_tick)


def get_timediffs(a, g, thr=THR):
    for tick in g:
        i = np.searchsorted(a, tick, side='left')
        try:
            if abs(tick - a[i - 1]) < abs(tick - a[i]):
                res = a[i - 1] - tick
            else:
                res = a[i] - tick
        except IndexError:
            res = a[i] - tick

        if thr[0] <= res <= thr[1]:
            yield res


def get_timediffs_double(a1, a2, g, thr1, thr2):
    for tick in g:
        i = np.searchsorted(a1, tick, side='left')
        j = np.searchsorted(a2, tick, side='left')

        try:
            if abs(tick - a1[i - 1]) < abs(tick - a1[i]):
                res1 = a1[i - 1] - tick
            else:
                res1 = a1[i] - tick
        except IndexError:
            res1 = a1[i] - tick

        try:
            if abs(tick - a2[j - 1]) < abs(tick - a2[j]):
                res2 = a2[j - 1] - tick
            else:
                res2 = a2[j] - tick
        except IndexError:
            res2 = a2[j] - tick

        if thr1[0] <= res1 <= thr1[1] and thr2[0] <= res2 <= thr2[1]:
            yield (res1, res2)


def timediffs_histo(arr, g, thr):
    dt = get_timediffs(arr, g, thr)
    bins = np.arange(*thr)
    vals = np.zeros_like(bins)

    for x in dt:
        vals[bins == x] += 1
    return (bins, vals)


#plt.hist(list(get_coincidences(r, g)), bins=np.arange(0,200), alpha=.5, label='r')


def get_all_timediffs(t, r, g):
    return (timediffs_histo(t, g, THR), timediffs_histo(r, g, THR))


def shapes(arr, n):
    return [np.shape(arr[arr % n == i])[0] for i in range(n)]


def get_odd_even_ratios(arr, num=50_000):
    dt = []
    for i in range(len(arr) // num):
        c = np.array(shapes(arr[num * i:num * (i + 1)], 2))
        dt.append(c[1] / c[0])
    return (dt)


def plot_oer(t, r, g):
    # oer = odd to even ratio

    dt = get_odd_even_ratios(t)
    dr = get_odd_even_ratios(r)
    dg = get_odd_even_ratios(g)

    plt.plot(np.linspace(0, 1, len(dr)), dr, label='refl')
    plt.plot(np.linspace(0, 1, len(dg)), dg, label='gate')
    plt.plot(np.linspace(0, 1, len(dt)), dt, label='tr')
    plt.legend()
    plt.title('odd to even ratios')
    plt.show()

def plot_timediffs(b_t, v_t, b_r, v_r,
                   b_tc, v_tc, b_rc, v_rc):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = iter(prop_cycle.by_key()['color'])
    
    c1 = next(colors)
    plt.bar(b_t, v_t, alpha=.5, color=c1)
    plt.bar(b_tc, v_tc, label="Time differences: transmitted - gate", alpha=.2, color=c1)
    c2 = next(colors)
    plt.bar(b_r, v_r, alpha=.5, color=c2)
    plt.bar(b_rc, v_rc, label="Time differences: reflected - gate", alpha=.2, color=c2)
    plt.xlabel('Time difference [integer multiples of 160ps]')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()


def get_statistics(bins, vals):

    def model(x, mean, std, const): return const * \
        norm(loc=mean, scale=std).pdf(x)

    popt, pcov = curve_fit(model, bins, vals, p0=[20, 10, 1000])

    # mean = np.average(bins, weights=vals)
    # var = np.average((bins - mean)**2, weights=vals)
    return (popt[0], popt[1])

    # return (mean, np.sqrt(var))


if __name__ == "__main__":
    ticks = get_ticks()

    T, R = get_all_timediffs(*ticks)
    t, r, g = ticks

    # m_t, s_t = get_statistics(*T)
    # m_r, s_r = get_statistics(*R)

    # std_multiplier = 5

    # thr_t = (int(round(m_t - s_t * std_multiplier)),
    #          int(round(m_t + s_t * std_multiplier)))
    # thr_r = (int(round(m_r - s_r * std_multiplier)),
    #          int(round(m_r + s_r * std_multiplier)))

    # b_tc, v_tc = timediffs_histo(t, g, THR)
    # b_rc, v_rc = timediffs_histo(r, g, THR)
    
    # b_t, v_t = timediffs_histo(t, g, thr_t)
    # b_r, v_r = timediffs_histo(r, g, thr_r)

    # td = get_timediffs_double(t, r, g, thr_t, thr_r)

    # N_G = len(g)
    # N_TG = np.sum(v_t)
    # N_RG = np.sum(v_r)
    # N_TRG = len(list(td))

    # print(f'{N_G=}')
    # print(f'{N_TG=}')
    # print(f'{N_RG=}')
    # print(f'{N_TRG=}')
    # print()
    # print(f'ratio r = {N_RG / N_G}')
    # print(f'ratio t = {N_TG / N_G}')
    # pass