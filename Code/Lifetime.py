from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
import pylab as pl
import scipy.optimize as spo
import lazy_property
from style import *
import os
cwd = os.getcwd()

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
	return p * 1e6 * e # in SI

def momentum_toMeV(p: float):
	return p * 1e-6 * c / e

def mass_toMeV(p: float):
	return p * 1e-6 * c**2 / e

def energy_toMeV(p: float):
	return p * 1e-6 / e


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

	@lazy_property.LazyProperty
	def labFrameTravel(self):
		return magnitude(self.dstarDecay-self.d0Decay) * 1e-3 # convert mm -> m

	@lazy_property.LazyProperty
	def pD0_t(self):
		pcomps_d0 = self.kp+self.pd # in MeV
		return magnitude(pcomps_d0[0:1])
	
	@lazy_property.LazyProperty
	def pPslow(self):
		return magnitude(self.ps)

	@lazy_property.LazyProperty
	def pPslow_t(self):
		return magnitude(self.ps[0:1])
	
	@lazy_property.LazyProperty
	def pD0(self):
		pcomps_d0 = self.kp+self.pd # in MeV
		return momentum_toSI(magnitude(pcomps_d0))

	@lazy_property.LazyProperty
	def pDstar(self):
		pcomps_dstar = self.kp+self.pd+self.ps # in MeV
		return momentum_toSI(magnitude(pcomps_dstar))

	@lazy_property.LazyProperty
	def daughterEnergy(self):
		p_pi = momentum_toSI(magnitude(self.pd))
		p_k = momentum_toSI(magnitude(self.kp))
		m_pi_si, m_k_si = mass_toSI(m_pi), mass_toSI(m_k)
		return np.sqrt((p_pi*c)**2 + m_pi_si**2 * c**4)   +   np.sqrt((p_k*c)**2 + m_k_si**2 * c**4)

	@lazy_property.LazyProperty
	def reconstructedD0Mass(self):
		E_de = self.daughterEnergy
		p_d0 = self.pD0
		return np.sqrt(E_de**2 - (p_d0*c)**2)/c**2

	@lazy_property.LazyProperty
	def gamma(self):
		p_d0 = self.pD0
		m_d0 = self.reconstructedD0Mass
		return p_d0 / (c * m_d0)

	@lazy_property.LazyProperty
	def decayTime(self):
		x = self.labFrameTravel
		m_d0 = self.reconstructedD0Mass
		p_d0 = self.pD0
		return x * m_d0 / p_d0

	@lazy_property.LazyProperty
	def dStarEnergy(self):
		E_d0 = self.daughterEnergy
		p_pislow = momentum_toSI(magnitude(self.ps))
		m_pi_si = mass_toSI(m_pi)
		return E_d0 + np.sqrt((p_pislow*c)**2 + m_pi_si**2 * c**4)

	@lazy_property.LazyProperty
	def reconstructedDstarMass(self):
		E = self.dStarEnergy
		p_dstar = self.pDstar
		return np.sqrt(E**2 - (p_dstar*c)**2)/c**2

	# in MeV
	@lazy_property.LazyProperty
	def massDiff_d0dstar(self):
		return mass_toMeV(self.reconstructedDstarMass - self.reconstructedD0Mass)

# reads a file and returns the D0 candidate events it lists
# - expects the file to have specific col titles, returns None if there is an error
def readFile(name: Text):
    with open(cwd+'/'+name, 'r') as csvfile:
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



def plotData(data):
	# mass dist
	masses = [mass_toMeV(d.reconstructedD0Mass) for d in data]
	newfig()
	pl.hist(masses, bins=100, histtype='step', fill=False)
	pl.xlabel(r'$D^0$ Mass / MeV/$c^2$')
	savefig('mass-dist')
	pl.close()

	# dstar mass dist
	ds_masses = [mass_toMeV(d.reconstructedDstarMass) for d in data]
	newfig()
	pl.hist(ds_masses, bins=100, histtype='step', fill=False)
	pl.xlabel(r'$D^{+*}$ Mass / MeV/$c^2$')
	savefig('dstar-mass-dist')
	pl.close()

	# mass difference dist
	mass_diffs = [x0 - x1 for x0, x1 in zip(ds_masses, masses)]
	newfig()
	pl.hist(mass_diffs, bins=100, histtype='step', fill=False)
	pl.xlabel(r'Mass difference / MeV/$c^2$')
	savefig('mass-diff-dist')
	pl.close()

	# gamma dist
	gammas = [d.gamma for d in data]
	newfig()
	pl.hist(gammas, bins=500)
	savefig('gamma-dist')
	pl.close()

	# travel dist
	trav = [d.labFrameTravel for d in data]
	newfig()
	pl.hist(trav, bins=500, range=(0, 0.04))
	savefig('trav-dist')
	pl.close()


	# decay time dist
	times = [d.decayTime*1e12 for d in data]
	
	newfig()
	pl.hist(times, bins=100, range=(0, 50))
	savefig('time-hist')
	pl.close()

	# decay time curve
	hist, bin_edges = np.histogram(times, bins=100, range=(0.1, 10))
	num_events = np.sum(hist)
	print(num_events, ' events')
	time = bin_edges[1:]

	newfig()
	pl.plot(time, hist, '-b')
	pl.xlabel(r'Decay time / ps')
	savefig('decay')
	pl.close()

	# decay time fitting
	po, po_cov = spo.curve_fit(lambda t, A, tau, c: A * np.exp(-t/tau)+c, time, hist, [num_events, 1.5, 0]) #TODO: error analysis, np.repeat(0.03, l-transition_idx), absolute_sigma=True)

	newfig()
	pl.plot(time, hist, '-b')
	pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-r')
	pl.xlabel(r'Decay time / ps')
	savefig('decay-fitted')
	pl.close()

	partial_lifetime = po[1]
	print('partial lifetime\t' + str(partial_lifetime) + ' ps')
	# print(hbar/(partial_lifetime*1e-12), hbar)
	# print('partial width   \t' + str(c**2 * 1e-6 * hbar/(partial_lifetime*1e-12) / e) + ' MeV/c2')


def plot_compare(accepted, rejected, prop, name, range=None):
	diffs_a, diffs_r = [getattr(d, prop) for d in accepted], [getattr(d, prop) for d in rejected]
	newfig()
	f, axarr = pl.subplots(2, sharex=True)
	axarr[0].hist(diffs_a, 100, facecolor='green', range=range)
	axarr[1].hist(diffs_r, 100, facecolor='red', range=range)
	savefig(name+'-compare')
	pl.close()


def background_fit(dm, bg_A, bg_p):
	return bg_A * (dm-m_pi)**bg_p
	
def signal_fit(dm, sig_A, sig_centre, sig_w):
	return sig_A * np.exp(-np.abs(dm-sig_centre)/sig_w)
# def signal_fit(dm, sig_A, sig_centre, sig_w):
# 	return sig_A * np.exp(-(dm-sig_centre)**2/sig_w)
	
def combined_fit(dm, bg_A, bg_p, sig_A, sig_centre, sig_w):
	return signal_fit(dm, sig_A, sig_centre, sig_w) + background_fit(dm, bg_A, bg_p)


# takes list of candidate events, cuts them by their mass diff
def cutEventSet_massDiff(events, width):

	initial = [20, 0.25, 220, 146, 2]
# 	initial = [10, 0.25, 220, 146, 2]

	diffs = [d.massDiff_d0dstar for d in events]
	hist, bin_edges = np.histogram(diffs, bins=300)
	masses = np.array([np.mean([d0, d1]) for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])
	bin_width = bin_edges[1] - bin_edges[0]

	po, po_cov = spo.curve_fit(combined_fit, masses, hist, initial)

	print('po-fit', po)
	newfig()
	pl.plot(masses, hist, '.g-')
	pl.plot(masses, signal_fit(masses, *po[2:]), '-r')
	pl.fill_between(masses, 0, background_fit(masses, *po[:2]), facecolor='blue', alpha=0.5)
	pl.xlabel(r'$\Delta m$ / GeV/$c^2$')
	savefig('cut-fitted')
	pl.close()

	# cut at 4 widths
	bg_A, bg_p, sig_A, sig_centre, sig_w = po
	range_low, range_up = sig_centre - sig_w*width, sig_centre + sig_w*width
	print('range', range_low, range_up)
	accepted = [event for event in events if range_low <= event.massDiff_d0dstar <= range_up]
	rejected = [event for event in events if not range_low <= event.massDiff_d0dstar <= range_up]
	return accepted, rejected, po, bin_width

def cut(accepted, rejected, cond):
	acc, rej = [], list(rejected)
	for a in accepted:
		if cond(a):
			acc.append(a)
		else:
			rej.append(a)

	return np.array(acc), np.array(rej)
