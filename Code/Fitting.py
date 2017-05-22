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

hist, bin_edges = np.histogram(times, bins=100, range=(0.1, 10))
time = bin_edges[1:]

ML = np.mean(times) #analytic mean lifetime by likelihood maximisation

first_lifetime_t = np.array([t for t in time if t < ML])
first_lifetime_hist = np.array(hist[:len(first_lifetime_t)])

Amp_int = first_lifetime_hist * np.exp(first_lifetime_t/ML)
Amp_avg = np.mean(Amp_int)

print('Amplitude', Amp_avg, 'Mean lifetime', ML)

newfig()
pl.plot(time, hist, '-b')
pl.plot(time, Amp_avg * np.exp(-time/ML), '-r')
pl.xlabel(r'Decay time / ps')
pl.savefig('decay-fitted-ML')
pl.close()
