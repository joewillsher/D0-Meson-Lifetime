import scipy.optimize as spo
import scipy.integrate as spi
import numpy as np
import scipy.special as sse

# constants
from scipy.constants import c, hbar, physical_constants
e = physical_constants['electron volt'][0]
m_pi, m_k = 139.57018, 493.677 # TODO: uncert 0.00035, 0.013 respectively

def convoluted_exponential(t, A, l, s, m):
	if s < 0 and m == 0:
		return 0
	return A* l/2 * np.exp(2*m + l * s**2 - 2*t) * sse.erfc((m + l * s**2 - t)/(2**0.5 * s))

def gaussian(t, A, s, m):
	return A * np.exp(-(t-m)**2/(2*s**2))

def double_gaussian(t, A, m, s1, s2, f):
	return A * (f * np.exp(-(t-m)**2/(2*s1**2)) + \
		(1-f) * np.exp(-(t-m)**2/(2*s2**2)))



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
	nsig = spi.quad(background_fit, range_low, range_up, args=(po[0], po[1], po[2]))[0]
	nb = spi.quad(background_fit, bg_kinematic_limit, max_dm, args=(po[0], po[1], po[2]))[0]
	na = nb-nsig
	wb = - na/nb
	print("Na", na)
	print("Nb", nb)
	print("wb", wb)
	return wb


def estimate_background(po, filtered, width):
	range_low, range_up = get_sig_range(po, width)
	bg_integral = spi.quad(background_fit, range_low, range_up, args=(po[0], po[1], po[2]))[0]
	sig_integral = spi.quad(signal_fit, range_low, range_up, args=(po[3], po[4], po[5], po[6], po[7]))[0]

	bg_fraction = bg_integral/(sig_integral + bg_integral)

	print("BACKGROUND EST", bg_integral, len(filtered), str(bg_fraction*100) + "%")
	return bg_integral, sig_integral, bg_fraction
