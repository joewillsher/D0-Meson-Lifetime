from Lifetime import *

# print(times)

filtered = np.load('fitting_FULLSET.npy')
po = np.load('fitting_AFTERPO.npy')
width = np.load('fitting_WIDTH.npy')[0]

# cut harshly on delta mass to remove bg, then get d star time
times = [event.dStarDecayTime*1e12 for event in filtered]

print(np.mean(times), np.std(times))


pl.hist(times, 40, range=(-3,3), histtype='step', fill=False)

# time = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])
# po, po_cov = spo.curve_fit(gaussian, time, hist, [max(hist), .2, 0])
# pl.plot(time, gaussian(time, *po), '-r')

print('res', po)
savefig('dstar-decay-time')
