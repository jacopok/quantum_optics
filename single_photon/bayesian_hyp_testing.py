import numpy as np
import matplotlib.pyplot as plt

from simulation import g_from_detections
from simulation import detections
from simulation import simulate_detections_classical
from simulation import simulate_detections_quantum
from scipy.stats import gaussian_kde
from tqdm import tqdm

from scipy.integrate import trapz


def plot_both_distributions(sample_size, rate, e_rate, ratio, N_gate):
    g_q = g_from_detections(
        *simulate_detections_quantum(sample_size, rate, e_rate, rate *
                                     ratio, e_rate * ratio, N_gate))
    g_c = g_from_detections(
        *simulate_detections_classical(sample_size, rate, e_rate, rate *
                                       ratio, e_rate * ratio, N_gate))
    bins = np.logspace(-3, .5, num=1000)
    bins[0] = 0
    # sample_size, probability_rate, error_rate, *_ = args

    # plt.semilogy(bins, gaussian_kde(g_q).pdf(bins), label='Quantum')
    # plt.semilogy(bins, gaussian_kde(g_c).pdf(bins), label='Classical')
    # plt.plot(bins, gaussian_kde(g_q).pdf(bins) / gaussian_kde(g_c).pdf(bins), label='Ratio')

    plt.hist(g_q, bins=bins, density=True, alpha=.5, label='Quantum')
    plt.hist(g_c, bins=bins, density=True, alpha=.5, label='Classical')
    plt.xscale('symlog', linthreshx=1e-5)
    plt.yscale('symlog')

    print(f'Quantum std: {np.std(g_q)}')
    print(f'Classical std: {np.std(g_c)}')
    plt.title(f'Pdfs with error {e_rate:.4f}, rate {rate:.4f}'
              f', sample size {sample_size}')
    plt.xlabel('$g^{(2)}$')
    plt.legend()
    plt.show(block=False)


def get_bayes_factor_parametric(measurement, sample_size, rate, e_rate, ratio,
                                parameters_prior):
    """the shape if parameters_prior should be (len(rate), len(e_rate))

    Should be called like:
    i, p, r, prior = get_bayes_factor_parametric(meas, 200, rate, e_rate, ratio, param_prior)
    
    where meas is a `measurement` named tuple
    
    
    after running the file 
    `parameter_estimates.py`
    
    then plot like 
    `plot_both_logpdfs(*r, p)`
    """
    N_1, N_2, N_12, N_gate = measurement

    measurement = np.array(measurement)
    to_sim = sample_size * parameters_prior.size * N_gate

    print(f'Need to compute {sample_size=} times '
          f'{parameters_prior.size=} times {N_gate=}')
    print(f'detections, which means 10 to the {np.log10(to_sim):.1f}')

    data_given_model_classical = np.zeros_like(parameters_prior)
    print('Estimating classical data-given-model')

    # Iterate over all possible values of the prior parameters,
    # for now we ignore the prior probability, we multiply by it later
    for i, r in enumerate(rate):
        print(f'{i+1} out of {len(rate)}')
        for j, e in tqdm(enumerate(e_rate)):

            # get the tuple of simulated detection numbers, according
            # to the parameters we have
            det_classical = simulate_detections_classical(
                sample_size, r, e, r * ratio, e, N_gate)

            try:
                # we use gaussian kernel density estimation in order
                # to estimate the probability density
                # given the parameters in this iterations
                # evaluated at the measurement
                likelihood = gaussian_kde(det_classical[:-1]).pdf(
                    measurement[:-1])
                # we compare only the first three parameters: they are the
                # only ones which can vary, the N_gate is constant by design
                # and the kde algorithm is not well-behaved for singular pdfs
            except (np.linalg.LinAlgError, ValueError):

                # if we get here it means the probability density
                # couldn't be estimated, so
                # we set the probability to zero
                likelihood = 0

            data_given_model_classical[i, j] = likelihood

    data_given_model_quantum = np.zeros_like(parameters_prior)
    print('Estimating quantum data-given-model')

    for i, r in enumerate(rate):
        print(f'{i+1} out of {len(rate)}')
        for j, e in tqdm(enumerate(e_rate)):
            det_quantum = simulate_detections_quantum(sample_size, r, e,
                                                      r * ratio, e, N_gate)
            try:
                likelihood = gaussian_kde(det_quantum[:-1]).pdf(
                    measurement[:-1])
            except (np.linalg.LinAlgError, ValueError):
                likelihood = 0
            data_given_model_quantum[i, j] = likelihood

    print('Integrating')
    rate_len, e_rate_len = np.shape(parameters_prior)
    integrand_classical = parameters_prior * data_given_model_classical
    integrand_quantum = parameters_prior * data_given_model_quantum

    if rate_len > 1:
        integrand_classical = trapz(y=integrand_classical, x=rate, axis=0)
        integrand_quantum = trapz(y=integrand_quantum, x=rate, axis=0)
    else:
        integrand_classical = integrand_classical[0]
        integrand_quantum = integrand_quantum[0]

    if e_rate_len > 1:
        # axis 0 is now what was axis 1 before
        likelihood_classical = trapz(y=integrand_classical, x=e_rate, axis=0)
        likelihood_quantum = trapz(y=integrand_quantum, x=e_rate, axis=0)
    else:
        likelihood_classical = integrand_classical[0]
        likelihood_quantum = integrand_quantum[0]

    from datetime import datetime
    now = datetime.now().isoformat()
    path = 'simulations/'
    to_save = {
        'dgm_classical': data_given_model_classical,
        'dgm_quantum': data_given_model_quantum,
        'rate': rate,
        'e_rate': e_rate,
        'param_prior': parameters_prior
    }

    with open(path + 'result_' + now + '.txt', 'w') as f:
        f.writelines('\n'.join([
            f'Classical pdf value: {likelihood_classical}',
            f'Quantum pdf value: {likelihood_quantum}',
            f'Ratio (log10): {-np.log10(likelihood_classical/likelihood_quantum)}',
            f'N_gate: {N_gate}', f'sample size: {sample_size}',
            f'parameter pdf shape: {parameters_prior.shape}'
            # f'g_measurement: {g_measurement}'
        ]))

    for name, x in to_save.items():
        np.save(path + name + '_' + now, x)

    return ((likelihood_classical, likelihood_quantum),
            (data_given_model_classical,
             data_given_model_quantum), (rate, e_rate), parameters_prior)


def uniform_dist_log(lower_log, upper_log, num=40):

    param = np.logspace(lower_log, min(upper_log, 0), base=np.e, num=num)

    dist = np.copy(param)
    normalization = trapz(y=dist, x=param)

    return (param, dist / normalization)


def uniform_dist(lower, upper, num=40):

    param = np.linspace(lower, min(upper, 1), num=num)

    dist = np.copy(param)
    normalization = trapz(y=dist, x=param)

    return (param, dist / normalization)


def plot_logpdf(rate, e_rate, pdf, ax, levels):

    cmap = plt.get_cmap('viridis')

    ax.set_facecolor(cmap.colors[0])

    log_e_rate = np.log(np.outer(np.ones_like(rate), e_rate))

    c = ax.contourf(rate,
                    np.log(e_rate), (np.log(pdf) + log_e_rate).T,
                    levels=levels,
                    cmap=cmap)
    ax.set_xlabel('detection rate')
    ax.set_ylabel('error rate (natural log)')
    return c


def plot_both_logpdfs(rate, e_rate, p):

    fig, axs = plt.subplots(1, 2)

    log_e_rate = np.log(np.outer(np.ones_like(rate), e_rate))

    max_p = max(np.max(np.log(p[0]) + log_e_rate),
                np.max(np.log(p[1]) + log_e_rate))
    levels = np.linspace(-400, max_p)

    c1 = plot_logpdf(rate, e_rate, p[0], axs[0], levels=levels)
    axs[0].set_title('Classical')
    fig.colorbar(c1, label='natural log of pdf', ax=axs[0])

    c2 = plot_logpdf(rate, e_rate, p[1], axs[1], levels)
    axs[1].set_title('Quantum')
    fig.colorbar(c2, label='natural log of pdf', ax=axs[1])

    fig.tight_layout()
    plt.show(block=False)
