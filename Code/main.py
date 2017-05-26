from style import *
# import Cuts
# import CutModel
# import Fitting


from Lifetime import *

bg_fraction = 0.003

times = np.array([t for t in np.load('TIMES.npy')])
bg_times = np.array([t for t in np.load('BG_TIMES.npy')])

newfig()
pl.hist(times, bins=100, range=(0, 50))
savefig('time-hist')
pl.close()

# decay time curve
hist, bin_edges = np.histogram(times, bins=120, range=(0., 10.))
bg_hist, bg_bin_edges = np.histogram(bg_times, bins=120, range=(0., 10.))

num_events = np.sum(hist)
num_bg = np.sum(bg_hist)
print(num_events, ' events')
# get the mean value in each bin
sy, _ = np.histogram(times, bins=bin_edges, weights=times)
time = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])
print(time)


bg_hist_normalised = bg_hist/num_bg * bg_fraction * num_events
subtracted_hist = hist - bg_hist_normalised
print(subtracted_hist, np.sum(subtracted_hist))
print(subtracted_hist, np.sum(subtracted_hist))
errors = [x*.999999 if x <= 1  else np.sqrt(x) for x in subtracted_hist-0.01]

# decay time fitting
po, po_cov = spo.curve_fit(lambda t, A, tau: A * np.exp(-t/tau), time, subtracted_hist, [num_events, 1.5])

po_conv, po_cov_conv = spo.curve_fit(convoluted_exponential, time, subtracted_hist, [num_events/2, .41, .05, 0.], errors, absolute_sigma=True)

newfig()
pl.semilogy(time, hist, '.g')
pl.semilogy(time, subtracted_hist, '.r')
pl.errorbar(time, subtracted_hist, yerr=errors, fmt=',r', capsize=0)
# 	if not is_latex:
# 		pl.semilogy(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-g')
# 		pl.semilogy(time, np.vectorize(lambda t: po[0] * np.exp(-t/np.mean(times)))(time), '-g')
pl.semilogy(time, convoluted_exponential(time, *po_conv), '-b')
pl.xlabel(r'Decay time [ps]')
savefig('decay-fitted')
pl.close()

newfig()
pl.plot(time, hist, '-g')
pl.plot(time, subtracted_hist, '-r')
pl.errorbar(time, subtracted_hist, yerr=errors, fmt=',r', capsize=0)
# 	if not is_latex:
# 		pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-g')
# 		pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/np.mean(times)))(time), '-g')
pl.plot(time, convoluted_exponential(time, *po_conv), '-b')
pl.xlabel(r'Decay time [ps]')
savefig('decay')
pl.close()


partial_lifetime = po[1]
mean_lifetime = np.mean(times)
print('convpo=', po_conv, '+-', np.sqrt(po_cov_conv[1][1]))
print('partial lifetime\t' + str(partial_lifetime) + ' ps', 'OR MEAN PL =', str(mean_lifetime)+'ps', 'OR CONV=', str(po_conv[1])+'ps')
