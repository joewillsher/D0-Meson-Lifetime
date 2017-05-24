from style import *
# import Cuts
# import CutModel
# import Fitting


from Lifetime import *

times = np.array([t for t in np.load('TIMES.npy')])

# decay time curve
hist, bin_edges = np.histogram(times, bins=120, range=(0., 8.))
num_events = len(times)
errors = error_hisogram(times, bin_edges, times)
print(errors)
print(num_events, ' events')
time = bin_edges[1:]

# decay time fitting
po, po_cov = spo.curve_fit(lambda t, A, tau: A * np.exp(-t/tau), time, hist, [num_events, 1.5]) #TODO: error analysis, np.repeat(0.03, l-transition_idx), absolute_sigma=True)

po_conv, po_cov_conv = spo.curve_fit(convoluted_exponential, time, hist, [num_events/2, .4, .1, 0.])

newfig()
pl.semilogy(time, hist, 'or')
pl.semilogy(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-r')
pl.semilogy(time, np.vectorize(lambda t: po[0] * np.exp(-t/np.mean(times)))(time), '-g')
pl.semilogy(time, convoluted_exponential(time, *po_conv), '-b')
pl.xlabel(r'Decay time [ps]')
savefig('decay-fitted')
pl.close()

newfig()
pl.plot(time, hist, '-k')
pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-r')
pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/np.mean(times)))(time), '-g')
pl.plot(time, convoluted_exponential(time, *po_conv), '-b')
pl.xlabel(r'Decay time [ps]')
savefig('decay')
pl.close()


partial_lifetime = po[1]
mean_lifetime = np.mean(times)
print('convpo=', po_conv)
print('partial lifetime\t' + str(partial_lifetime) + ' ps', 'OR MEAN PL =', str(mean_lifetime)+'ps', 'OR CONV=', str(po_conv[1])+'ps')
