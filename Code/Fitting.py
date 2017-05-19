from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
import pylab as pl
import scipy.optimize as spo
import lazy_property
from Lifetime import * #e, m_pi, Position, Momentum, magnitude, momentum_toSI, mass_toSI, energy_toSI, momentum_toMeV, mass_toMeV, energy_toMeV, CandidateEvent, labFrameTravel, pD0_t, pPslow, pD0, pDstar, daughterEnergy, reconstructedD0Mass, gamma, decayTime, dStarEnergy, reconstructedDstarMass, massDiff_d0dstar,
import Cuts
from Cuts import filtered

times = [d.decayTime*1e12 for d in filtered if 10e-12 >= d.decayTime] #list

newfig()
pl.hist(times, bins=100, range=(0, 10))
savefig('time-hist-fit')
pl.close()


range_t=np.linspace(0.15, 10.14, 100) #time range

N = len(times)
print(N)
T= sum(times)
ML = T/N #analytic mean lifetime by likelihood maximisation
print(ML)
print(T)

P = [] #probability of decay up to that point

for t in range_t:
    p =np.exp(-t/ML)
    P.append(p)

pl.plot(range_t, P, '-b')
pl.savefig('fitting_1.png')
pl.close()
