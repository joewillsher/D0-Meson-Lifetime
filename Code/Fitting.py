@@ -1,61 +0,0 @@
from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
import pylab as pl
import scipy.optimize as spo
import lazy_property
from Lifetime import readFile #e, m_pi, Position, Momentum, magnitude, momentum_toSI, mass_toSI, energy_toSI, momentum_toMeV, mass_toMeV, energy_toMeV, CandidateEvent, labFrameTravel, pD0_t, pPslow, pD0, pDstar, daughterEnergy, reconstructedD0Mass, gamma, decayTime, dStarEnergy, reconstructedDstarMass, massDiff_d0dstar,

data = readFile('np.txt')
times = [d.decayTime*1e12 for d in data] #list

flag =True

range_t=np.linspace(0.15, 10.14, 100) #time range
decay= []
for t in range_t:
    Decay_proba=np.exp(-t/4.7)
    decay.append(Decay_proba)
    #if t > 9.3 and t <9.5 and flag :
        #print('9.4', t, Decay_proba, '/n')
        #flag = False
i =np.linspace(0, 99, 100, dtype=int)
#test=np.random.RandomState.exponential(scale=0.25, size=100)
#print('test', test)

#histogram_values = [ decay[j] * range_t[j] for j in i]
histogram_values=[]
for j in i:
    A=int(decay[j])
    t_i=range_t[j]
    h=1
    while h <= A :
        h = h+1
        histogram_values.append(t_i)
print('histo', histogram_values)

print('length', len(histogram_values))
print('histogram values', histogram_values)
pl.plot(range_t, decay)
pl.savefig('test.png')
pl.grid(True)
pl.close()

N = len(range_t)
print(N)
T= sum(histogram_values)
ML = (1/N)*T #analytic mean lifetime by likelihood maximisation
print(ML)
print(T)

P = [] #probability of decay up to that point

#for t in range_t:
    #p =np.exp(-t/ML)
    #P.append(p)

pl.plot(range_t, P, '-b')
pl.savefig('fitting_1.png')
pl.close()
