from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, physical_constants
import pylab as pl
import scipy.optimize as spo

# constants
e = physical_constants['electron volt'][0]
m_pi, m_k = 139.57018, 493.677 # TODO: uncert 0.00035, 0.013 respectively

# position 3-vector, units mm
Position = np.array
# momentum 3-vector, units MeV/c
Momentum = np.array
# Momentum = namedtuple('Momentum', ['x', 'y', 'z'])

def magnitude(v: Position):
	return np.sqrt(np.sum(v**2))

def momentum_toSI(p: float):
	return p * 1e6 * e / c # in SI

def mass_toSI(p: float):
	return p * 1e6 * e / c**2 # in SI

def energy_toSI(p: float):
	return p * 1e6 # in SI


# Keep all data about the type together in a type, so vals for an event are stored next to each other on the heap.
class CandidateEvent(object):

	def __init__(self, interaction: Position, dstarDecay: Position, d0Decay: Position, kp: Momentum, pd: Momentum, ps: Momentum):
		self.interaction = interaction
		self.dstarDecay = dstarDecay
		self.d0Decay = d0Decay
		self.kp = kp
		self.pd = pd
		self.ps = ps

	def __repr__(self):
		str = super(CandidateEvent, self).__repr__() + "\n"
		str += "Dstar_OWNPV\t" + np.array_str(self.interaction) + "\n"
		str += "Dstar_ENDVERTEX\t" + np.array_str(self.dstarDecay) + "\n"
		str += "D_ENDVERTEX\t" + np.array_str(self.d0Decay) + "\n"
		str += "K\t\t" + np.array_str(self.kp) + "\n"
		str += "Pd\t\t" + np.array_str(self.pd) + "\n"
		str += "Ps\t\t" + np.array_str(self.ps) + "\n"
		return str

	def labFrameTravel(self):
		return magnitude(self.dstarDecay-self.d0Decay) * 1e-3 # convert mm -> m
	
	def pD0(self):
		pcomps_d0 = self.kp+self.pd # in MeV
		return momentum_toSI(magnitude(pcomps_d0))
	
	def daughterEnergy(self):
		p_pi = momentum_toSI(magnitude(self.pd))
		p_k = momentum_toSI(magnitude(self.kp))
		m_pi_si, m_k_si = mass_toSI(m_pi), mass_toSI(m_k)
		print('p daughter=', p_pi, p_k)
		return np.sqrt((p_pi*c)**2 + m_pi_si**2 * c**4)   +   np.sqrt((p_k*c)**2 + m_k_si**2 * c**4)

	def reconstructedD0Mass(self):
		E_de = self.daughterEnergy()
		p_d0 = self.pD0()
		print('de', E_de, p_d0)
		return np.sqrt(E_de**2 - (p_d0*c)**2)/c**2
	
	def gamma(self):
		p_d0 = self.pD0()
		m_d0 = self.reconstructedD0Mass()
		return p_d0 / (c * m_d0)
	
	def decayTime(self):
		x = self.labFrameTravel()
		m_d0 = self.reconstructedD0Mass()
		p_d0 = self.pD0()
		return x * m_d0 / p_d0


# reads a file and returns the D0 candidate events it lists
# - expects the file to have specific col titles, returns None if there is an error
def readFile(name: Text):
    with open(name, 'r') as csvfile:
    	# get the rows from the file
        rows = csv.reader(csvfile, delimiter=' ', quotechar='|') #creates array of array: each array is a row.

        # take the first element of the generator as the header
        header = next(rows)
        print(header)
        # store the candidates here
        cands = []

		# ignore the coordinate, remove the last '_X'/'_PX' part in the header name
        header_raw_names = ['_'.join(name.split('_')[:-1]) for name in header[::3]]
        order = ['Dstar_OWNPV', 'Dstar_ENDVERTEX', 'D_ENDVERTEX', 'K', 'Pd', 'Ps']
        print('names', header_raw_names)
        # get the indicies of these params in the row
        elementIdxs = [header_raw_names.index(x) for x in order]
        print(elementIdxs)

        for row in rows:
        	# reshape row into several 3-vectors
        	nums = [float(x) for x in row]
        	data = np.reshape(np.array(nums), (len(nums)//3, int(3))) 
        	cand = CandidateEvent(data[elementIdxs[0]], data[elementIdxs[1]], data[elementIdxs[2]], data[elementIdxs[3]], data[elementIdxs[4]], data[elementIdxs[5]])
        	cands.append(cand)

        return cands


#get data
data = readFile('np.txt')
d = data[0]
print(d)
mass, time = d.reconstructedD0Mass(), d.decayTime()
print('mass', mass, mass*1e-6*c**2 / e, time, time*1e15)

# mass dist
masses = [d.reconstructedD0Mass() for d in data]
pl.hist(masses, bins=500)
pl.savefig('mass-dist.png')

# gamma dist
gammas = [d.gamma() for d in data]
pl.hist(gammas, bins=500)
pl.savefig('gamma-dist.png')


# decay time dist
times = [d.decayTime()*1e15 for d in data]

pl.hist(times, bins=500, range=(0, 10000))
pl.savefig('time-hist.png')

# decay time curve
hist, bin_edges = np.histogram(times, bins=500, range=(0, 10000))
bin_width = bin_edges[1]-bin_edges[0]

cum = np.cumsum(hist)*bin_width
time = bin_edges[1:]

pl.plot(time, cum)
pl.savefig('decay.png')

# decay time fitting
po, po_cov = spo.curve_fit(lambda t, A, tau, c: A * np.exp(-t/tau) + c, time, cum, [1, 400, 0]) #TODO: error analysis, np.repeat(0.03, l-transition_idx), absolute_sigma=True)
print(po)

