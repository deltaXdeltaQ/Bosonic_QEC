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
# Specify global parameters
Dim = 100
SumNum = 6

# %%
# Predefine some operators
# PositionOper = position(Dim)
# MomentumOper = momentum(Dim)
# IdentityOper  = identity(Dim)
S_position = (1j*np.sqrt(2*np.pi)*position(Dim)).expm()
S_momentum = (- 1j*np.sqrt(2*np.pi)*momentum(Dim)).expm()

SignalState = coherent(Dim, 2/np.sqrt(2))
GKP1 = CreateGKPState(Dim, SumNum)

# Define the noisy operators
DeltaPos = 0.5
PositionNoiseOper = (- 1j*DeltaPos*momentum(Dim)).expm()


# Define sum and difference gate
SumGate = (-1j*tensor(position(Dim), identity(Dim))*tensor(identity(Dim), momentum(Dim))).expm()
DifferenceGate = (1j*tensor(position(Dim), identity(Dim))*tensor(identity(Dim), momentum(Dim))).expm()

# %%
Dim = 100
SumNum = 10
GKP1 = CreateGKPState(Dim, SumNum)
PlotWignerState(GKP1)


# %%
# QECC process:
# Encoding
InitialState = tensor(SignalState, GKP1)
#InitialDM = InitialState.ptrace(0)
EncodedState = SumGate(InitialState)
#EncodedDM = EncodedState*EncodedState.dag()

# Noise channel
StateAfterNoisyChannel = tensor(PositionNoiseOper, identity(Dim))*EncodedState

# Decoding
DecodedState = DifferenceGate*StateAfterNoisyChannel
FinalSignalDM = DecodedState.ptrace(0)
FinalAncillaDM = DecodedState.ptrace(1)


# %%
PlotWignerDM(GKP1)
PlotWignerDM(FinalAncillaDM)

PlotWignerDM(SignalState)
PlotWignerDM(FinalSignalDM)

# %%
AncillaPositionMeasurementResult = np.angle((FinalAncillaDM*S_position).tr())/np.sqrt(2*np.pi)
SignalPositionMeasurementResult = np.angle((SignalState*SignalState.dag()`*S_position).tr())/np.sqrt(2*np.pi)
print('Measured Ancilla Position:', AncillaPositionMeasurementResult)
print('Measured Sinal Position:', SignalPositionMeasurementResult)


# %%
tensor([GKP1,GKP1,GKP1])

# %%




