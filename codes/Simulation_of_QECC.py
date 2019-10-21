# %%
import numpy as np 
from qutip import *
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import cm
%matplotlib inline

# %%
#Define a function to plot the wigner function of a pure state
def Plot_wigner_state(Plot_state):
	xvec = np.linspace(-4,4,100)
	Plot_dm = Plot_state*Plot_state.dag()
	Plot_wigner = wigner(Plot_dm, xvec,  xvec)
	Plot_map = wigner_cmap(Plot_wigner)
	plt.figure()
	plt.contourf(xvec, xvec, Plot_wigner, 100)
	plt.colorbar()
	plt.show()


# %%
#Create GKP code
Dim = 500 #System dimension
Sum_num = 20 # Summation number of GKP state
GKP_state = 0
Vac = basis(Dim,0)
for i in np.arange(-Sum_num/2,Sum_num/2 + 1,1):
	alpha = np.sqrt(2*np.pi)/np.sqrt(2)*i
	Pos_state = (displace(Dim, alpha)*squeeze(Dim, 2)*Vac).unit()
	GKP_state = GKP_state + Pos_state
GKP_state = GKP_state.unit()
#GKP_state = (displace(Dim, 4/np.sqrt(2))*squeeze(Dim, 3)*Vac).unit()
#Plot_wigner_state(GKP_state)

Plot_wigner_state(GKP_state)

# %%
'''
#Plot the wigner function of the generated GKP state
xvec = np.linspace(-4,4,100)
GKP_dm = GKP_state*GKP_state.dag()
GKP_wigner = wigner(GKP_dm, xvec,  xvec)

GKP_map = wigner_cmap(GKP_wigner)
plt.figure()
plt.contourf(xvec, xvec, GKP_wigner, 100)
plt.colorbar()
plt.show()
'''

#Define stablizer operators
S1 = (1j*np.sqrt(2*np.pi)*position(Dim)).expm()
S2 = (- 1j*np.sqrt(2*np.pi)*momentum(Dim)).expm()

# Noise operators
delta_pos = 0.2
Position_Noise = (- 1j*delta_pos*momentum(Dim)).expm()

#matrix_histogram(Position_Noise)
'''
plt.figure()
plt.contourf(Position_Noise.full())
plt.show()
'''

GKP_state_final = Position_Noise*GKP_state
Error_syndrome1 = (GKP_state_final.dag()*S1*GKP_state_final).full()
Error_syndrome1 = np.angle(Error_syndrome1[0,0])
Error_syndrome2 = (GKP_state_final.dag()*S2*GKP_state_final).full()
Error_syndrome2 = np.angle(Error_syndrome2[0,0])

#Plot_wigner_state(GKP_state_final - GKP_state)

print('Error Syndrome1:', Error_syndrome1/(np.sqrt(2*np.pi)), 'Error Syndrome2:', Error_syndrome2/(np.sqrt(2*np.pi)))
print(GKP_state.dag()*S2*GKP_state)
