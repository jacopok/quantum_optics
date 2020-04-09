def get_all_timediffs(names, resolution = RESOLUTION):
  diffs = []
  for name in names:
    t = get_ticks(name)
    dt = np.ediff1d(t)
    diffs.append(dt)

  return (u.Quantity(np.hstack(diffs), resolution))

def get_all_ticks(names, resolution = RESOLUTION):
  total_ticks = []
  for name in names:
    t = get_ticks(name)
    total_ticks.append(t)

  return (u.Quantity(np.hstack(total_ticks), resolution))

def moment(b, v, n):
  m = np.average(b, weights=v)
  if (n == 1):
    return(m)
  return (np.sum((b - m)**n * v))

def analyze(b, v):
  moments = {'mean': 1, 'variance': 2, 'skewness': 3, 'kurtosis': 4}
  for m, num in moments.items():
    print(m, ' = ', f'{moment(v, b, num):.3f}')


if __name__ == '__main__':
  # timediffs = {}
  # timediffs['coherent'] = get_all_timediffs(COHERENT_NAMES)
  # timediffs['thermal'] = get_all_timediffs(THERMAL_NAMES)

  # bins = {}
  # values = {}

  # time_unit = u.us
  # max_time = 50

  # for x in timediffs:
  #   times = timediffs[x].to(time_unit).value
  #   mask = times < max_time
  #   values[x], bins[x], _ = plt.hist(times[mask], label=x, bins=200, alpha=.6, density=True)
  # plt.xlabel(f'Time differences [{str(time_unit)}]')
  # plt.xlim(0, max_time)
  # plt.legend()
  # plt.show()

  # for x in timediffs:
  #   print(x)
  #   b = bins[x]
  #   analyze((b[1:]+b[:-1])/2., values[x])
