from Fitting import *

# ti = np.linspace(-4., 10., 1000)
# 
# newfig()
# pl.plot(ti, convoluted_exponential(ti, 1, .41, 0.8, 0), '-b')
# pl.xlabel(r'Decay time [ps]')
# savefig('d-BACKGROUNDSIUBTR')
# pl.close()


full_set = np.load('fitting_FULLSET.npy')
after_po = np.load('fitting_AFTERPO.npy')
deltamass_peak_width = np.load('fitting_WIDTH.npy')[0]

maximum_likelyhood_exp_fit(full_set, after_po, deltamass_peak_width)

