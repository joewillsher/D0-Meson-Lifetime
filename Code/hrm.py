from Lifetime import *
import pylab as pl

#get data
data = readFile('np.txt')
d = data[0]
print(d)
mass, time = d.reconstructedD0Mass, d.decayTime
print('mass', mass, mass*1e-6*c**2 / e, time, time*1e15)


def cut(accepted, rejected, cond):
	acc, rej = [], list(rejected)
	for a in accepted:
		if cond(a):
			acc.append(a)
		else:
			rej.append(a)

	return np.array(acc), np.array(rej)


# cut on mass diff
filtered, rejected, po = cutEventSet_massDiff(data)
bg, sig = po[:2], po[2:]

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

accepted_t = [f.decayTime*1e12 for f in filtered if 1e-13 <= f.decayTime <= 1e-11]
tau_m = np.mean(accepted_t) # mean lifetime
tau_b = np.mean(rejected_t) # mean lifetime
print(tau_m, tau_b)

plot_compare(filtered, rejected, 'decayTime', 'decay-BEFORE', (0,10e-12))

# filtered, rejected = cut(filtered, rejected, lambda d: 2500 <= d.pD0_t <= 20000)
# filtered, rejected = cut(filtered, rejected, lambda d: 300 <= d.pPslow_t)
# filtered, rejected = cut(filtered, rejected, lambda d: 2000 <= d.pPslow)
# plot_compare(filtered, rejected, 'decayTime', 'decay-AFTER', (0,10e-12))

newfig()
pl.semilogy(time, hist, 'or')
pl.semilogy(time, po[0] * np.exp(-np.array(time)/tau_b)+po[2], '-r')
# pl.semilogy(time, np.max(accepted_t) * np.exp(-np.array(time)/tau_m), '-g')
savefig('fit-reject')
pl.close()


diffs = [d.massDiff_d0dstar for d in filtered]
hist, bin_edges = np.histogram(diffs, bins=10)
dms = np.array([np.mean([d0, d1]) for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])
diff_events = np.array([[x for x in filtered if d0 <= x.massDiff_d0dstar < d1] for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])

print(diff_events)

def expected_signal_number(dm):
	return signal_fit(dm, *sig)

def expected_background_number(dm):
	return background_fit(dm, *bg)

N_b = expected_background_number(dms)
N_s = expected_signal_number(dms)
N = N_b + N_s
tau_m_event = np.array([np.mean([d.decayTime*1e12 for d in event if 1e-13 <= d.decayTime <= 1e-11]) for event in diff_events])

a_b = (N * tau_m_event - N_b * tau_b) / N_s


print(N_b)
print(N_s)
print(N)
print(tau_m_event)
taus = tau_b * (N_b / N_s) * a_b
weights = N_s/N
print(weights)
print(taus, np.average(taus[2:-2],weights=weights[2:-2]))



"""
diffs = [d.massDiff_d0dstar for d in data]
hist, bin_edges = np.histogram(diffs, bins=200)
dms = np.array([np.mean([d0, d1]) for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])
diff_events = np.array([[x for x in filtered if d0 <= x.massDiff_d0dstar < d1] for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])


print(diff_events)

def expected_signal_number(dm):
	return signal_fit(dm, *sig)

def expected_background_number(dm):
	return background_fit(dm, *bg)

N_b = expected_background_number(dms)
N_s = expected_signal_number(dms)
N = N_b + N_s
tau_m_event = np.array([np.mean([d.decayTime*1e12 for d in event if 1e-13 <= d.decayTime <= 1e-11]) for event in diff_events])

fig, ax = newfig()
pl.semilogy([f.massDiff_d0dstar for f in filtered if f.decayTime < 20e-12], [f.decayTime*1e12 for f in filtered if f.decayTime < 20e-12], ',g')
pl.semilogy([f.massDiff_d0dstar for f in rejected if f.decayTime < 20e-12], [f.decayTime*1e12 for f in rejected if f.decayTime < 20e-12], ',r')
pl.semilogy(dms, tau_m_event, '-b')
ax.set_ylim(ymin=0.1, ymax=20)
savefig('decayTime-average-correlation')
pl.close()
"""













from Lifetime import *
import pylab as pl

#get data
data = readFile('np.txt')
d = data[0]
print(d)
mass, time = d.reconstructedD0Mass, d.decayTime
print('mass', mass, mass*1e-6*c**2 / e, time, time*1e15)


def cut(accepted, rejected, cond):
	acc, rej = [], list(rejected)
	for a in accepted:
		if cond(a):
			acc.append(a)
		else:
			rej.append(a)

	return np.array(acc), np.array(rej)


# cut on mass diff
filtered, rejected, po = cutEventSet_massDiff(data)
bg, sig = po[:2], po[2:]

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

accepted_t = [f.decayTime*1e12 for f in filtered if 1e-13 <= f.decayTime <= 1e-11]
tau_m = np.mean(accepted_t) # mean lifetime
tau_b = np.mean(rejected_t) # mean lifetime
print(tau_m, tau_b)

plot_compare(filtered, rejected, 'decayTime', 'decay-BEFORE', (0,10e-12))

# filtered, rejected = cut(filtered, rejected, lambda d: 2500 <= d.pD0_t <= 20000)
# filtered, rejected = cut(filtered, rejected, lambda d: 300 <= d.pPslow_t)
# filtered, rejected = cut(filtered, rejected, lambda d: 2000 <= d.pPslow)
# plot_compare(filtered, rejected, 'decayTime', 'decay-AFTER', (0,10e-12))

newfig()
pl.semilogy(time, hist, 'or')
pl.semilogy(time, po[0] * np.exp(-np.array(time)/tau_b)+po[2], '-r')
# pl.semilogy(time, np.max(accepted_t) * np.exp(-np.array(time)/tau_m), '-g')
savefig('fit-reject')
pl.close()


sig_centre, sig_w = po[3], po[4]
width = 1.
range_low, range_up = sig_centre - sig_w*width, sig_centre + sig_w*width

diffs = [d.massDiff_d0dstar for d in filtered if range_low <= d.massDiff_d0dstar <= range_up]

diff_events = np.array([[x for x in filtered if d0 <= x.massDiff_d0dstar < d1] for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])

print(diff_events)

def expected_signal_number(dm):
	return signal_fit(dm, *sig)

def expected_background_number(dm):
	return background_fit(dm, *bg)

N_b = scipy.quad(expected_background_number, range_low, range_up)
N_s = scipy.quad(expected_signal_number, range_low, range_up)
N = N_b + N_s
tau_m_event = np.mean([d.decayTime*1e12 for d in event if 1e-13 <= d.decayTime <= 1e-11])

a_b = (N * tau_m_event - N_b * tau_b) / N_s


print(N_b)
print(N_s)
print(N)
print(tau_m_event)
taus = tau_b * (N_b / N_s) * a_b
weights = N_s/N
print(weights)
print(taus, np.average(taus))



"""
diffs = [d.massDiff_d0dstar for d in data]
hist, bin_edges = np.histogram(diffs, bins=200)
dms = np.array([np.mean([d0, d1]) for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])
diff_events = np.array([[x for x in filtered if d0 <= x.massDiff_d0dstar < d1] for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])


print(diff_events)

def expected_signal_number(dm):
	return signal_fit(dm, *sig)

def expected_background_number(dm):
	return background_fit(dm, *bg)

N_b = expected_background_number(dms)
N_s = expected_signal_number(dms)
N = N_b + N_s
tau_m_event = np.array([np.mean([d.decayTime*1e12 for d in event if 1e-13 <= d.decayTime <= 1e-11]) for event in diff_events])

fig, ax = newfig()
pl.semilogy([f.massDiff_d0dstar for f in filtered if f.decayTime < 20e-12], [f.decayTime*1e12 for f in filtered if f.decayTime < 20e-12], ',g')
pl.semilogy([f.massDiff_d0dstar for f in rejected if f.decayTime < 20e-12], [f.decayTime*1e12 for f in rejected if f.decayTime < 20e-12], ',r')
pl.semilogy(dms, tau_m_event, '-b')
ax.set_ylim(ymin=0.1, ymax=20)
savefig('decayTime-average-correlation')
pl.close()
"""
