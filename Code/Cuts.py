from Lifetime import *

#get data
data = readFile('np.txt')
d = data[0]
print(d)
mass, time = d.reconstructedD0Mass, d.decayTime
print('mass', mass, mass*1e-6*c**2 / e, time, time*1e15)





filtered = data
# cut on mass diff
filtered, rejected = cutEventSet_massDiff(filtered)


newfig()
pl.plot([f.massDiff_d0dstar for f in filtered], [f.pD0_t for f in filtered], ',g')
pl.plot([f.massDiff_d0dstar for f in rejected], [f.pD0_t for f in rejected], ',r')
savefig('dm-pD0_t-correleation')
pl.close()

newfig()
pl.plot([f.massDiff_d0dstar for f in filtered], [f.pPslow for f in filtered], ',g')
pl.plot([f.massDiff_d0dstar for f in rejected], [f.pPslow for f in rejected], ',r')
savefig('dm-pPslow-correleation')
pl.close()

newfig()
pl.plot([f.massDiff_d0dstar for f in filtered], [f.pPslow_t for f in filtered], ',g')
pl.plot([f.massDiff_d0dstar for f in rejected], [f.pPslow_t for f in rejected], ',r')
savefig('dm-pPslow_t-correleation')
pl.close()

newfig()
pl.plot([f.massDiff_d0dstar for f in filtered if f.decayTime < 10e-12], [f.decayTime for f in filtered if f.decayTime < 10e-12], ',g')
pl.plot([f.massDiff_d0dstar for f in rejected if f.decayTime < 10e-12], [f.decayTime for f in rejected if f.decayTime < 10e-12], ',r')
savefig('dm-decayTime-correleation')
pl.close()


plot_compare(filtered, rejected, 'decayTime', 'decayTime')
plot_compare(filtered, rejected, 'pD0_t', 'pd0-t')
plot_compare(filtered, rejected, 'pPslow', 'pslow')
plot_compare(filtered, rejected, 'pPslow_t', 'pslow-t')

# cut on transverse momentum
filtered = [d for d in filtered if 2500 <= d.pD0_t <= 20000]
filtered = [d for d in filtered if 2000 <= d.pPslow]
filtered = [d for d in filtered if 300 <= d.pPslow_t]

# remove width
d0_c, dstar_c = 1865., 2010.
width = 20.
filtered = [event for event in filtered if (d0_c-width) <= mass_toMeV(event.reconstructedD0Mass) <= (d0_c+width) and (dstar_c-width) <= mass_toMeV(event.reconstructedDstarMass) <= (dstar_c+width)]


# filtered = [d for d in filtered if d.pPslow >= 1.3e3]

plotData(filtered)

# mass dist
offs = [d.pPslow for d in filtered]
pl.hist(offs, bins=100, histtype='step', fill=False)
pl.savefig('offsets.png')
pl.close()
