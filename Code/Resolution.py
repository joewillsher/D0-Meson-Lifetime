from Lifetime import *

# print(times)

data = np.load('fitting_FULLSET.npy')

wb = -0.2939171976029101
range_low, range_up = 142.414041989, 149.059198523

times = [d.dStarDecayTime*1e12 for d in data]
# bg_times = [d.decayTime*1e12 for d in data if not range_low <= d.massDiff_d0dstar <= range_up]
weights = [(1 if range_low <= d.massDiff_d0dstar <= range_up else wb) for d in data]

time_range, bin_num = (-10, 10), 120

# decay time curve
hist, bin_edges = np.histogram(times, bins=bin_num, range=time_range, weights=weights)
hist_unw, _ = np.histogram(times, bins=bin_num, range=time_range)

sy = np.histogram(times, bins=bin_edges, weights=times)[0]
time = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])

newfig()
pl.semilogy(time, hist, '.g')
pl.semilogy(time, hist_unw, '.b')
pl.xlabel(r'Decay time [ps]')
savefig('dstar-decay-time')
pl.close()

print(np.mean(times), np.std(times))

