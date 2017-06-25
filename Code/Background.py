import scipy.optimize as spo
import scipy.integrate as spi
import numpy as np
import scipy.special as sse
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

# constants
from scipy.constants import c, hbar, physical_constants
e = physical_constants['electron volt'][0]
m_pi, m_k = 139.57018, 493.677 # TODO: uncert 0.00035, 0.013 respectively

def convoluted_exponential(t, A, tau, s):
	return A * 1/(2*tau) * np.exp(1/tau * (s**2/(2*tau) - t)) * (1- sse.erf((s**2 / tau - t)/(2**0.5 * s)))

def gaussian(t, A, s, m):
	return A * np.exp(-(t-m)**2/(2*s**2))

def double_gaussian(t, A, m, s1, s2, f):
	return A * (f * np.exp(-(t-m)**2/(2*s1**2)) + \
		(1-f) * np.exp(-(t-m)**2/(2*s2**2)))


def normalisation_const(pdf, range, args):
	i = spi.quad(pdf, range[0], range[1], args=args)[0]
	return 1/i

def background_fit(dm, bg_A, bg_p, bg_m):
	return bg_A * (dm-bg_m)**bg_p


signal_fit = double_gaussian
# 2.07302440e+01   2.98646045e-01   1.77728941e+03   1.45465825e+02		7.79647227e+00  -6.76180865e-01   4.09293926e+00]
def combined_fit(dm, bg_A, bg_p, bg_m, sig_A, sig_centre, sig_w1, sig_w2, f):
	return signal_fit(dm, sig_A, sig_centre, sig_w1, sig_w2, f) + background_fit(dm, bg_A, bg_p, bg_m)



def get_sig_range(po, width):
	bg_A, bg_p, bg_m, sig_A, sig_centre, sig_w1, sig_w2, f = po
	sig_w = max(sig_w1, sig_w2)
	range_low, range_up = sig_centre - sig_w*width, sig_centre + sig_w*width
	return range_low, range_up


def calculate_weight(po, filtered, range_low, range_up):
	sig_centre, sig_w = po[3], po[4]
	bg_kinematic_limit, max_dm = po[2], 165
	na = spi.quad(background_fit, range_low, range_up, args=(po[0], po[1], po[2]))[0]
	nb = spi.quad(background_fit, bg_kinematic_limit, max_dm, args=(po[0], po[1], po[2]))[0]-na
	wb = - na/nb
	print("Na", na)
	print("Nb", nb)
	print("wb", wb)
	return wb


def estimate_background(po, filtered, bin_width, width):
	range_low, range_up = get_sig_range(po, width)
	bg_integral = spi.quad(background_fit, range_low, range_up, args=(po[0], po[1], po[2]))[0]/bin_width
	sig_integral = spi.quad(signal_fit, range_low, range_up, args=(po[3], po[4], po[5], po[6], po[7]))[0]/bin_width

	bg_fraction = bg_integral/(sig_integral + bg_integral)

	print("BACKGROUND EST", bg_integral, len(filtered), str(bg_fraction*100) + "%")
	return bg_integral, sig_integral, bg_fraction


class Record(object):
	def __init__(self, name, data):
		self.name = name
		self.data = data

write_list = []

def write_out():
	with open("data.txt", "w") as text_file:
		for d in write_list:
			text_file.write("%s=%s\n" % (d.name, d.data))

def add_val(name, val, round_to=1):
	write_list.append(Record(name, np.round(val, round_to)))

def add_int(name, d):
	write_list.append(Record(name, locale.format("%d", d, grouping=True)))
