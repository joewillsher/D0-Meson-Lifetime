from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, physical_constants
import pylab as pl
import scipy.optimize as spo
e = physical_constants['electron volt'][0]

# position 3-vector, units mm
Position = np.array
# momentum 3-vector, units MeV/c
Momentum = np.array
# Momentum = namedtuple('Momentum', ['x', 'y', 'z'])

def magnitude(v: Position):
	return np.sqrt(np.sum(v**2))


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
	
	def absoluteDecayTime(self):
		# p in a 3 axes is conserved in `D0 --> pi + K` decay
		pcomps_d0 = self.kp+self.pd # in MeV
		print(pcomps_d0)
		p_d0 = magnitude(pcomps_d0) * 1e6 * e / c # in SI
		m0_n = 1864.84 # MeV/c2
		m0 = m0_n * 1e6 * e / c**2 # convert MeV/c* -> kg
		# un-boosted time in the particle's frame
		x = self.labFrameTravel()  
		tp = m0 * x / p_d0
		print(x, p_d0, m0, tp)
		return tp
		
		

# reads a file and returns the D0 candidate events it lists
# - expects the file to have specific col titles, returns None if there is an error
def readFile(name: Text):
    with open(name, 'r') as csvfile:
    	# get the rows from the file
        rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
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
        	data = np.reshape(np.array(nums), (len(nums)/3, 3))
        	cand = CandidateEvent(data[elementIdxs[0]], data[elementIdxs[1]], data[elementIdxs[2]], data[elementIdxs[3]], data[elementIdxs[4]], data[elementIdxs[5]])
        	cands.append(cand)
        	
        return cands
        

def getLifetime():
	data = readFile('np.txt')
	print(data)
	times = [d.absoluteDecayTime()*1e15 for d in data]
	
	hist, bin_edges = np.histogram(times, bins=500, range=(0, 10000))
	pl.hist(times, bins=500, range=(0, 10000))
	pl.savefig('hist.png')
	bin_width = bin_edges[1]-bin_edges[0]
	
	cum = np.cumsum(hist)*bin_width
	time = bin_edges[1:]
	
	pl.plot(time, cum)
	pl.savefig('graph.png')
	
	po, po_cov = spo.curve_fit(lambda t, A, tau, c: A * np.exp(-t/tau) + c, time, cum, [1, 400, 0]) #TODO: error analysis, np.repeat(0.03, l-transition_idx), absolute_sigma=True)
	
	print('po', po)
	
	print(hist, len(cum))

getLifetime()
