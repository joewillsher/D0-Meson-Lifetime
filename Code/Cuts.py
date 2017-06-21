from Lifetime import *

def cut(accepted, rejected, cond):
	acc, rej = [], list(rejected)
	for a in accepted:
		if cond(a):
			acc.append(a)
		else:
			rej.append(a)

	return np.array(acc), np.array(rej)

#get data
data = readFile('np.txt' if '--full-set' in sys.argv else 'np-short.txt')
d = data[0]
print(d)
mass, time = d.reconstructedD0Mass, d.decayTime
print('mass', mass, mass*1e-6*c**2 / e, time, time*1e15)

# cut on mass diff
width = 3.
signal_region, background_sidebands, po_fullset, bin_width = cutEventSet_massDiff(data, width)
filtered = data
rejected = []
print(len(data))
bg_integral, sig_integral, bg_fraction = estimate_background(po_fullset, filtered, width)



if not '--no-plot' in sys.argv:
	print('plot')

	massDiff_plot(data, ext_name='-KK', fit=False, methodName='reconstructedD0Mass_kk', range=None)
	massDiff_plot(data, ext_name='-PP', fit=False, methodName='reconstructedD0Mass_pp', range=None)

	newfig()
	pl.plot([f.massDiff_d0dstar for f in signal_region], [f.pD0_t for f in signal_region], ',g')
	pl.plot([f.massDiff_d0dstar for f in background_sidebands], [f.pD0_t for f in background_sidebands], ',r')
	savefig('dm-pD0_t-correlation')
	pl.close()

	newfig()
	pl.plot([f.massDiff_d0dstar for f in signal_region], [f.pPslow for f in signal_region], ',g')
	pl.plot([f.massDiff_d0dstar for f in background_sidebands], [f.pPslow for f in background_sidebands], ',r')
	savefig('dm-pPslow-correlation')
	pl.close()

	newfig()
	pl.plot([f.massDiff_d0dstar for f in signal_region], [f.pPslow_t for f in signal_region], ',g')
	pl.plot([f.massDiff_d0dstar for f in background_sidebands], [f.pPslow_t for f in background_sidebands], ',r')
	savefig('dm-pPslow_t-correlation')
	pl.close()

	fig, ax = newfig()
	pl.semilogy([f.massDiff_d0dstar for f in signal_region if f.decayTime < 20e-12], [f.decayTime*1e12 for f in signal_region if f.decayTime < 20e-12], ',g')
	pl.semilogy([f.massDiff_d0dstar for f in background_sidebands if f.decayTime < 20e-12], [f.decayTime*1e12 for f in background_sidebands if f.decayTime < 20e-12], ',r')
	ax.set_ylim(ymin=0.1, ymax=20)
	savefig('dm-decayTime-correlation')
	pl.close()

	newfig()
	pl.plot([f.massDiff_d0dstar for f in signal_region], [f.d0IP_log for f in signal_region], ',g')
	pl.plot([f.massDiff_d0dstar for f in background_sidebands], [f.d0IP_log for f in background_sidebands], ',r')
	savefig('dm-d0IP_log-correlation')
	pl.close()


	plot_compare(signal_region, background_sidebands, 'decayTime', 'decayTime', range=(0, 10e-12), \
		label=r'Decay time $t$ [ps]')
	plot_compare(signal_region, background_sidebands, 'pD0_t', 'pd0-t', \
		label=r'$D^0$ Transverse momentum $p_T$ [GeV / c]')
	plot_compare(signal_region, background_sidebands, 'pPslow', 'pslow', \
		label=r'Slow $\pi$ momentum p [GeVp_T / c]')
	plot_compare(signal_region, background_sidebands, 'pPslow_t', 'pslow-t', \
		label=r'Slow $\pi$ transverse momentum $p_T$ [GeV / c]')
	plot_compare(signal_region, background_sidebands, 'pDstar_t', 'dstar-t', \
		label=r'$D^{*+}$ transverse momentum $p_T$ [GeV / c]')
	plot_compare(signal_region, background_sidebands, 'pk_t', 'pk-t', \
		label=r'Daughter $k$ transverse momentum $p_T$ [GeV / c]')
	plot_compare(signal_region, background_sidebands, 'pp_t', 'pp-t', \
		label=r'Daughter $\pi$ transverse momentum $p_T$ [GeV / c]')

	plot_compare(signal_region, background_sidebands, 'd0IP_log', 'd0-impact-parameter', \
		label=r'$\log{\left(IP_{D^0} / \mathrm{mm}\right)}$')
	plot_compare(signal_region, background_sidebands, 'kIP_log', 'k-impact-parameter', \
		label=r'$\log{\left(IP_{k} / \mathrm{mm}\right)}$')
	plot_compare(signal_region, background_sidebands, 'pIP_log', 'p-impact-parameter', \
		label=r'$\log{\left(IP_{\pi} / \mathrm{mm}\right)}$')
	plot_compare(signal_region, background_sidebands, 'psIP_log', 'ps-impact-parameter', \
		label=r'$\log{\left(IP_{\pi_s} / \mathrm{mm}\right)}$')
	
	plot_compare(signal_region, background_sidebands, 's_z', 's_z', range=(-200, 200), \
		label=r'$s_z$ [mm]')
	plot_compare(signal_region, background_sidebands, 'costheta', 'costheta', range=(.999,1), \
		label=r'$\cos{\theta}$')


print('cut')
# filtered, rejected = cut(filtered, rejected, lambda d: 2500 <= d.pD0_t) # 4500, 2500
# filtered, rejected = cut(filtered, rejected, lambda d: 1400 <= d.pDstar_t < 20000) # 2500, 1400
# filtered, rejected = cut(filtered, rejected, lambda d: 200 <= d.pPslow_t < 2500) # 300, 200
# filtered, rejected = cut(filtered, rejected, lambda d: 700 <= d.pk_t) # 1000, 700
# filtered, rejected = cut(filtered, rejected, lambda d: 700 <= d.pp_t) # 1000, 700

# filtered, rejected = cut(filtered, rejected, lambda d: -5 <= d.d0IP_log <= -2.5)
# filtered, rejected = cut(filtered, rejected, lambda d: -4 <= d.kIP_log <= -0.1)
# filtered, rejected = cut(filtered, rejected, lambda d: -4 <= d.pIP_log <= -0.1)
# filtered, rejected = cut(filtered, rejected, lambda d: -4 <= d.psIP_log <= -1.1)

# filtered, rejected = cut(filtered, rejected, lambda d:  20 <= d.s_z <= 120)
# filtered, rejected = cut(filtered, rejected, lambda d:  .9995 <= d.costheta or d.costheta <= -.9995)

print('cut-done')

after_po, after_bin_width = massDiff_plot(filtered, ext_name='after', bg_ratio=.01)


# remove width
d0_c, dstar_c = 1865., 2010.
meson_mass_width = 20.
filtered = [event for event in filtered if (d0_c - meson_mass_width) <= mass_toMeV(event.reconstructedD0Mass) <= (d0_c + meson_mass_width) and (dstar_c - meson_mass_width) <= mass_toMeV(event.reconstructedDstarMass) <= (dstar_c + meson_mass_width)]



# massDiff_plot(filtered, 'AFTER', 0)
plotData(filtered)	
calculateLifetime(filtered, background_sidebands, after_po, width)

# mass dist
newfig()
offs = [d.pPslow for d in filtered]
pl.hist(offs, bins=100, histtype='step', fill=False)
savefig('offsets')
pl.close()
