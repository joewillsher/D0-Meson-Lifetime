from Lifetime import *
import pylab as pl

#get data
data = readFile('np.txt')
d = data[0]
print(d)
mass, time = d.reconstructedD0Mass, d.decayTime
print('mass', mass, mass*1e-6*c**2 / e, time, time*1e15)




# cut on mass diff
filtered, rejected, po_cut_data, data_bin_width = cutEventSet_massDiff(data, 4.)
bg, sig = po_cut_data[:2], po_cut_data[2:]

# fit the initial cut's rejected data

# decay time dist
rejected_t = [d.decayTime*1e12 for d in rejected if 1e-13 <= d.decayTime <= 1e-11]
# decay time curve
hist, bin_edges = np.histogram(rejected_t, bins=100, range=(0.1, 10))
time = bin_edges[1:]
# rejected time fitting
po, po_cov = spo.curve_fit(lambda t, A, tau, c: A * np.exp(-t/tau)+c, time, hist, [len(rejected), 1.5, 0])
tau_b = po[1]
print(po)

# accepted_t = [f.decayTime*1e12 for f in filtered if 1e-13 <= f.decayTime <= 1e-11]
# tau_m = np.mean(accepted_t) # mean lifetime
# tau_b = np.mean(rejected_t) # mean lifetime
# print(tau_m, tau_b)

plot_compare(filtered, rejected, 'decayTime', 'decay-BEFORE', (0,10e-12))

filtered, rejected = cut(filtered, rejected, lambda d: 2500 <= d.pD0_t <= 20000)
filtered, rejected = cut(filtered, rejected, lambda d: 140 <= d.pPslow_t)
filtered, rejected = cut(filtered, rejected, lambda d: 2000 <= d.pPslow)
plot_compare(filtered, rejected, 'decayTime', 'decay-AFTER', (0,10e-12))


newfig()
pl.semilogy(time, hist, 'or')
pl.semilogy(time, po[0] * np.exp(-np.array(time)/tau_b)+po[2], '-r')
# pl.semilogy(time, np.max(accepted_t) * np.exp(-np.array(time)/tau_m), '-g')
savefig('fit-reject')
pl.close()




def expected_signal_number(dm):
	return signal_fit(dm, *sig)/data_bin_width

def expected_background_number(dm):
	return background_fit(dm, *bg)/data_bin_width

sig_centre, sig_w = po_cut_data[3], po_cut_data[4]
# PARAMS
width = .8
offset = +.5*width
num_bins = 10
range_low, range_up = offset+sig_centre - sig_w*width, offset+sig_centre + sig_w*width

print('RANGE', range_low, range_up)

# all events we consider in the range
events = [d for d in filtered if range_low <= d.massDiff_d0dstar <= range_up and 1e-13 <= d.decayTime <= 1e-11]
print('events', len(events))
# array of md's
diffs = [d.massDiff_d0dstar for d in events]
# hist of dm's
hist, bin_edges = np.histogram(diffs, bins=num_bins)
# central dm vals
dms = np.array([np.mean([d0, d1]) for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])
dm_width = dms[1]-dms[0]
# the vals corresponding to the
diff_events = np.array([[x for x in events if d0 <= x.massDiff_d0dstar < d1] for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])

N_b = expected_background_number(dms)*dm_width
N_s = expected_signal_number(dms)*dm_width
N = N_b + N_s
tau_m_event = []
for dm_mean, events in zip(dms, diff_events):
	hist, bin_edges = np.histogram([e.decayTime*1e12 for e in events], bins=20)
	time = bin_edges[1:]
	po, po_conv = spo.curve_fit(lambda t, A, tau, c: A * np.exp(-t/tau)+c, time, hist, [len(rejected), 1.5, 0])
	tau_m_event.append(po[1])

tau_m_event = np.array(tau_m_event)

a_b = (N * tau_m_event - N_b * tau_b) / N_s
weights = N / N_b

print(N_b, sum(N_b))
print(N_s, sum(N_s))
print(tau_m_event)
print(tau_b)
print(a_b)
print(weights)
print(np.average(a_b, weights=weights))


# fig, ax = newfig()
# pl.semilogy([f.massDiff_d0dstar for f in events], [f.decayTime*1e12 for f in events], ',g')
# pl.semilogy([f.massDiff_d0dstar for f in rejected if f.decayTime < 20e-12], [f.decayTime*1e12 for f in rejected if f.decayTime < 20e-12], ',r')
# pl.semilogy(dms, tau_m_event, '-b')
# ax.set_ylim(ymin=0.1, ymax=20)
# savefig('decayTime-average-correlation')
# pl.close()
# 
