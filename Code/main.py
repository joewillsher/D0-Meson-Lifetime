from Fitting import *

# ti = np.linspace(-4., 10., 1000)
# 
# newfig()
# pl.plot(ti, convoluted_exponential(ti, 1, .41, 0.8, 0), '-b')
# pl.xlabel(r'Decay time [ps]')
# savefig('d-BACKGROUNDSIUBTR')
# pl.close()


# full_set = np.load('fitting_FULLSET.npy')
# after_po = np.load('fitting_AFTERPO.npy')
# deltamass_peak_width = np.load('fitting_WIDTH.npy')[0]
# 
# maximum_likelyhood_exp_fit(full_set, after_po, deltamass_peak_width)

data = [d for d in np.load('fitting_FULLSET.npy') if 0 <= d.decayTime < 10e-12]

wb = -0.2939171976029101
bg_fraction = 0.022
range_low, range_up = 142.414041989, 149.059198523

times = [d.decayTime*1e12 for d in data]
bg_times = [np.abs(d.decayTime)*1e12 for d in data if not range_low <= d.massDiff_d0dstar <= range_up]
weights = [(1 if range_low <= d.massDiff_d0dstar <= range_up else wb) for d in data]

time_range, bin_num = (0, 10), 120

# decay time curve
hist, bin_edges = np.histogram(times, bins=bin_num, range=time_range, weights=weights)
hist_unw, _ = np.histogram(times, bins=bin_num, range=time_range)
hist_bg = np.histogram(bg_times, bins=bin_num, range=time_range, normed=True)[0] * np.sum(hist) * bg_fraction

sy = np.histogram(times, bins=bin_edges, weights=times)[0]
time = bin_edges[1:]
# time = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])
errors = [x*.9999999999 if x <= 1 else np.sqrt(x) for x in hist-.0000000001]


newfig()
pl.semilogy(time, hist, '.g')
pl.semilogy(time, hist_bg, '.k')
pl.errorbar(time, hist, yerr=errors, fmt=',r', capsize=0)
pl.xlabel(r'Decay time [ps]')
savefig('decay-fitted')
pl.close()

newfig()
pl.plot(time, hist, ',g')
pl.plot(time, hist_unw, '.b')
pl.errorbar(time, hist, yerr=errors, fmt=',r', capsize=0)
pl.xlabel(r'Decay time [ps]')
savefig('decay')	
pl.close()
