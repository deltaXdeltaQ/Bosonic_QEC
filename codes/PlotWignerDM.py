import numpy as np 
from qutip import *
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import cm

def PlotWignerDM(PlotDM):
	xvec = np.linspace(-4,4,100)
	Plot_wigner = wigner(PlotDM, xvec,  xvec)
	Plot_map = wigner_cmap(Plot_wigner)
	plt.figure()
	plt.contourf(xvec, xvec, Plot_wigner, 100)
	plt.colorbar()
	plt.show()