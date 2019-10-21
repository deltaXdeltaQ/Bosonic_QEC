from qutip import *
import numpy as np 


def CreateGKPState(Dims, Sum_num):
	GKP_state = 0
	Vac = basis(Dims,0)
	for i in np.arange(-Sum_num/2,Sum_num/2 + 1,1):
		alpha = np.sqrt(2*np.pi)/np.sqrt(2)*i
		Pos_state = (displace(Dims, alpha)*squeeze(Dims, 2)*Vac).unit()
		GKP_state = GKP_state + Pos_state
	GKP_state = GKP_state.unit()
	return GKP_state

