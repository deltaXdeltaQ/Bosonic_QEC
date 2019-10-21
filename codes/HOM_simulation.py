# %%
import sys
sys.path.append('/Users/qian/Documents/Research/JiangGroupProjects/QEC_with_bosinic_code/codes')

# %%
import numpy as np 
from qutip import *
import matplotlib.pyplot as plt 
from CreateGKPState import *
from PlotWignerDM import *
from PlotWignerState import *
import copy

# %%
Dim = 30
SumNum = 4

CatState1 = (coherent(Dim, 1/np.sqrt(2)) + coherent(Dim, - 1/np.sqrt(2))).unit()
CatState2 = (coherent(Dim, 1/np.sqrt(2)) - coherent(Dim, - 1/np.sqrt(2))).unit()
GKPState1 = CreateGKPState(Dim, SumNum)
FockState1 = basis(Dim,1)
VacuumState = basis(Dim, 0)


CoherentState1 = coherent(Dim, 1.4/np.sqrt(2))

CoincidenceCount = tensor(num(Dim), num(Dim))

Mode1 = GKPState1
Mode2 = GKPState1
SystemState = tensor(Mode1, Mode2)

print('Before BS:', 'Average photon number in mode 1:', (Mode1*Mode1.dag()*num(Dim)).tr(), 'Average photon number in mode 2:', (Mode2*Mode2.dag()*num(Dim)).tr())
print('coincidence count before BS:', (SystemState*SystemState.dag()*CoincidenceCount).tr())

#BS = (1j*np.pi/4*(tensor(create(Dim), identity(Dim))*tensor(identity(Dim), destroy(Dim)) + tensor(destroy(Dim), identity(Dim))*tensor(identity(Dim), create(Dim)))).expm()
BS = (1j*np.pi/4*(tensor(create(Dim), destroy(Dim))+ tensor(destroy(Dim), create(Dim)))).expm()


StateAfterBS = BS*SystemState
DMAfterBS = StateAfterBS*StateAfterBS.dag()


Mode1DM = DMAfterBS.ptrace(0)
Mode2DM = DMAfterBS.ptrace(1)

print('After BS:','Average photon number in mode 1:', (Mode1DM*num(Dim)).tr(), 'Average photon number in mode 2:', (Mode2DM*num(Dim)).tr())
print('coincidence count after BS:', (CoincidenceCount*DMAfterBS).tr())

