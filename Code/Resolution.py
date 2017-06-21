from Lifetime import *

# print(times)

_data = np.load('fitting_FULLSET.npy')

data = [d for d in _data]

wb = -0.2939171976029101
range_low, range_up = 142.414041989, 149.059198523

times = [d.dStarDecayTime*1e12 for d in data]
weights = [(1 if range_low <= d.massDiff_d0dstar <= range_up else wb) for d in data]

time_range, bin_num = (-2, 0), 120

# decay time curve
hist, bin_edges = np.histogram(times, bins=bin_num, range=time_range, weights=weights)
hist_unw, _ = np.histogram(times, bins=bin_num, range=time_range)

sy = np.histogram(times, bins=bin_edges, weights=times)[0]
time = bin_edges[:-1]
# errors = [x*.9999999999 if x <= 1 else np.sqrt(x) for x in hist-.0000000001]

po_conv, po_cov_conv = spo.curve_fit(gaussian, time, hist, [1000, .8, 0])#, errors, absolute_sigma=True)

newfig()
pl.semilogy(time, hist, '.g')
pl.semilogy(time, gaussian(time, *po_conv), '-g')
pl.semilogy(time, hist_unw, '.b')
pl.xlabel(r'Decay time [ps]')
savefig('dstar-decay-time')
pl.close()

print(np.mean(times), np.std(times), po_conv)

