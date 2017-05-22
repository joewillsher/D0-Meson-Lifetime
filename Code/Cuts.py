from Lifetime import *

#get data
data = readFile('np.txt')
d = data[0]
print(d)
mass, time = d.reconstructedD0Mass, d.decayTime
print('mass', mass, mass*1e-6*c**2 / e, time, time*1e15)


filtered = data
# cut on mass diff
filtered, rejected, po, bin_width = cutEventSet_massDiff(filtered, 1.)


newfig()
pl.plot([f.massDiff_d0dstar for f in filtered], [f.pD0_t for f in filtered], ',g')
pl.plot([f.massDiff_d0dstar for f in rejected], [f.pD0_t for f in rejected], ',r')
savefig('dm-pD0_t-correlation')
pl.close()

newfig()
pl.plot([f.massDiff_d0dstar for f in filtered], [f.pPslow for f in filtered], ',g')
pl.plot([f.massDiff_d0dstar for f in rejected], [f.pPslow for f in rejected], ',r')
savefig('dm-pPslow-correlation')
pl.close()

newfig()
pl.plot([f.massDiff_d0dstar for f in filtered], [f.pPslow_t for f in filtered], ',g')
pl.plot([f.massDiff_d0dstar for f in rejected], [f.pPslow_t for f in rejected], ',r')
savefig('dm-pPslow_t-correlation')
pl.close()

fig, ax = newfig()
pl.semilogy([f.massDiff_d0dstar for f in filtered if f.decayTime < 20e-12], [f.decayTime*1e12 for f in filtered if f.decayTime < 20e-12], ',g')
pl.semilogy([f.massDiff_d0dstar for f in rejected if f.decayTime < 20e-12], [f.decayTime*1e12 for f in rejected if f.decayTime < 20e-12], ',r')
ax.set_ylim(ymin=0.1, ymax=20)
savefig('dm-decayTime-correlation')
pl.close()


plot_compare(filtered, rejected, 'decayTime', 'decayTime', (0, 10e-12))
plot_compare(filtered, rejected, 'pD0_t', 'pd0-t')
plot_compare(filtered, rejected, 'pPslow', 'pslow')
plot_compare(filtered, rejected, 'pPslow_t', 'pslow-t')
plot_compare(filtered, rejected, 'pDstar_t', 'dstar-t')
plot_compare(filtered, rejected, 'pk_t', 'pk-t')
plot_compare(filtered, rejected, 'pp_t', 'pp-t')
plot_compare(filtered, rejected, 'd0IP_log', 'd0-impact-parameter')
plot_compare(filtered, rejected, 'kIP_log', 'k-impact-parameter')
plot_compare(filtered, rejected, 'pIP_log', 'p-impact-parameter')
plot_compare(filtered, rejected, 'psIP_log', 'ps-impact-parameter')
plot_compare(filtered, rejected, 's_z', 's_z', (-200, 200))
plot_compare(filtered, rejected, 'costheta', 'costheta', (.6,1))


filtered, rejected = cut(filtered, rejected, lambda d: 3500 <= d.pD0_t)
filtered, rejected = cut(filtered, rejected, lambda d: 2500 <= d.pDstar_t)
filtered, rejected = cut(filtered, rejected, lambda d: 200 <= d.pPslow_t)
filtered, rejected = cut(filtered, rejected, lambda d: 500 <= d.pk_t)
filtered, rejected = cut(filtered, rejected, lambda d: 500 <= d.pp_t)

filtered, rejected = cut(filtered, rejected, lambda d: -2 <= d.d0IP_log <= 1)
filtered, rejected = cut(filtered, rejected, lambda d: -1 <= d.kIP_log <= 3)
filtered, rejected = cut(filtered, rejected, lambda d: -1 <= d.pIP_log <= 3)
filtered, rejected = cut(filtered, rejected, lambda d: -1 <= d.psIP_log <= 2)

filtered, rejected = cut(filtered, rejected, lambda d:  10 <= d.s_z <= 120)
filtered, rejected = cut(filtered, rejected, lambda d:  0.95 <= d.costheta)



# remove width
d0_c, dstar_c = 1865., 2010.
width = 20.
filtered = [event for event in filtered if (d0_c-width) <= mass_toMeV(event.reconstructedD0Mass) <= (d0_c+width) and (dstar_c-width) <= mass_toMeV(event.reconstructedDstarMass) <= (dstar_c+width)]



# filtered = [d for d in filtered if d.pPslow >= 1.3e3]

plotData(filtered)

# mass dist
newfig()
offs = [d.pPslow for d in filtered]
pl.hist(offs, bins=100, histtype='step', fill=False)
savefig('offsets')
pl.close()
