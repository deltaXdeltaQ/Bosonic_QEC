import numpy as np 
from qutip import *
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import cm

def PlotWignerState(PlotState):
	xvec = np.linspace(-4,4,100)
	Plot_dm = PlotState*PlotState.dag()
	Plot_wigner = wigner(Plot_dm, xvec,  xvec)
	Plot_map = wigner_cmap(Plot_wigner)
	plt.figure()
	plt.xlabel('p', fontsize = 16)
	plt.ylabel('q', fontsize = 16)
	plt.contourf(xvec, xvec, Plot_wigner, 100)
	plt.colorbar()
	plt.show()