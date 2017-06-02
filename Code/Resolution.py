from Cuts import *

times = [d.dStarDecayTime*1e12 for d in filtered]

# print(times)
print(np.mean(times))

pl.hist(times, 40, range=(-3,3), histtype='step', fill=False)
# sy, _ = np.histogram(times, bins=bin_edges, weights=times)
# time = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])

# po, po_cov = spo.curve_fit(gaussian, time, hist, [max(hist), .2, 0])
# pl.plot(time, gaussian(time, *po), '-r')

print('res', po)
savefig('dstar-decay-time')
