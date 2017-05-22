from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
import pylab as pl
import scipy.optimize as spo
import lazy_property
from Lifetime import *
from Cuts import filtered

times = [d.decayTime*1e12 for d in filtered if 10e-12 >= d.decayTime] #decay times considered

#estimations of fitting parameters:
hist, bin_edges = np.histogram(times, bins=100, range=(0.1, 10))
time = bin_edges[1:]

ML = np.mean(times) #analytic mean lifetime by likelihood maximisation

first_lifetime_t = np.array([t for t in time if t < ML]) #selects the area of data to look at for fitting
first_lifetime_hist = np.array(hist[:len(first_lifetime_t)]) #corresponding histogram values

Amp_int = first_lifetime_hist * np.exp(first_lifetime_t/ML)
Amp_avg = np.mean(Amp_int) #average amplitude

print('Amplitude', Amp_avg, 'Mean lifetime', ML)

#plots fit + data on the same graph
newfig()
pl.plot(time, hist, '-b')
pl.plot(time, Amp_avg * np.exp(-time/ML), '-r')
pl.xlabel(r'Decay time / ps')
pl.savefig('decay-fitted-ML')
pl.close()

#chi squared test:
expected_value = Amp_avg * np.exp(-time/ML)
observed_value = hist
diff_sqrd = (observed_value - expected_value)**2
chi_int = diff_sqrd/expected_value

pl.plot(time, chi_int)
pl.savefig('chi_squared vs time')

chi_squared = sum(chi_int)
meaan_chi_squared = np.mean(chi_int)
#print('expected_value', expected_value, len(expected_value),'observed_value', observed_value, len(observed_value), 'diff_sqrd', diff_sqrd, len(diff_sqrd), 'chi_int', chi_int, len(chi_int))
print('chi_squared', chi_squared, 'meaan_chi_squared', meaan_chi_squared)
#Chi_2_result = sum(chi_2)
#print('chi_2', chi_2, 'Chi_2_result', Chi_2_result)
