from matplotlib import pylab as plt
def init_rc_params():
	plt.rcParams['font.family'] = 'serif'
	plt.rcParams['font.serif'] = 'FreeSerif' # for Ubuntu
	plt.rcParams['lines.linewidth'] = 2
	plt.rcParams['xtick.labelsize'] = 24
	plt.rcParams['ytick.labelsize'] = 24
	plt.rcParams['legend.fontsize'] = 24
	plt.rcParams['figure.figsize'] = (20, 15)

