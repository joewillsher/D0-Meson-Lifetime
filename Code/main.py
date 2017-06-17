from Fitting import *


full_set = np.load('fitting_FULLSET.npy')
after_po = np.load('fitting_AFTERPO.npy')
deltamass_peak_width = np.load('fitting_WIDTH.npy')[0]

maximum_likelyhood_exp_fit(full_set, after_po, deltamass_peak_width)

