from Cuts import *

# print(times)


range_low, range_up = get_sig_range(po, .4)
# cut harshly on delta mass to remove bg, then get d star time
times = [d.dStarDecayTime*1e12 for event in filtered if range_low <= event.massDiff_d0dstar <= range_up]

print(np.mean(times), np.std(times))


pl.hist(times, 40, range=(-3,3), histtype='step', fill=False)

# time = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])
# po, po_cov = spo.curve_fit(gaussian, time, hist, [max(hist), .2, 0])
# pl.plot(time, gaussian(time, *po), '-r')

print('res', po)
savefig('dstar-decay-time')
