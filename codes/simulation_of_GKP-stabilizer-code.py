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
from GKPStabilizerCode import *
import copy

# %%
### Define excitation loss operator
gamma = 0.1
E0 = identity(Dim) - gamma/2*num(Dim)
E1 = np.sqrt(gamma)*destroy(Dim)

# %%
def CreateSumGate(i,j, GKPRepititionNum):
	TempOperList1 = [identity(Dim) for num in range(GKPRepititionNum + 1)]
	TempOperList1[i] = position(Dim)

	TempOperList2 = [identity(Dim) for num in range(GKPRepititionNum + 1)]
	TempOperList2[j] = momentum(Dim)

	return (-1j*tensor(TempOperList1)*tensor(TempOperList2)).expm()

def CreateDifferenceGate(i,j, GKPRepititionNum):
	TempOperList1 = [identity(Dim) for num in range(GKPRepititionNum + 1)]
	TempOperList1[i] = position(Dim)

	TempOperList2 = [identity(Dim) for num in range(GKPRepititionNum + 1)]
	TempOperList2[j] = momentum(Dim)

	return (1j*tensor(TempOperList1)*tensor(TempOperList2)).expm()

# %%
# Specify global parameters
Dim = 20
SumNum = 4
GKPRepititionNum = 1

S_position = (1j*np.sqrt(2*np.pi)*position(Dim)).expm()
S_momentum = (- 1j*np.sqrt(2*np.pi)*momentum(Dim)).expm()

SignalState = coherent(Dim, 2/np.sqrt(2))
GKP1 = CreateGKPState(Dim, SumNum)

# Define the noisy operators
DeltaPos = 0.3
PositionNoiseOper = (- 1j*DeltaPos*momentum(Dim)).expm()
#NoiseOperList = [tensor(PositionNoiseOper, identity(Dim), identity(Dim))]
NoiseOperList = [tensor(PositionNoiseOper, identity(Dim))]

# EncodingOper = CreateSumGate(0,1,GKPRepititionNum)*CreateSumGate(0,2,GKPRepititionNum)
# DecodingOper = CreateDifferenceGate(0,1,GKPRepititionNum)*CreateDifferenceGate(0,2,GKPRepititionNum)
EncodingOper = CreateSumGate(0,1,GKPRepititionNum)
DecodingOper = CreateDifferenceGate(0,1,GKPRepititionNum)

# %%
GKPStabilizerCode1 = GKPStabilizerCode(EncodingOperator_input = EncodingOper, DecodingOperator_input = DecodingOper, OscillatorDim_input= Dim, GKPRepititionNum_input = GKPRepititionNum, GKPSumNum_input= SumNum)

GKPStabilizerCode1.InputDataMode(SignalState)

GKPStabilizerCode1.Encode()

GKPStabilizerCode1.ThroughNoisyChannel(NoiseOperList)

GKPStabilizerCode1.Decode()

print('Measurement result:', GKPStabilizerCode1.Measure())

# %%
#PlotWignerDM(GKPStabilizerCode1.SystemState.ptrace(1))

SystemDM = GKPStabilizerCode1.SystemState*GKPStabilizerCode1.SystemState.dag()

for i in range(GKPStabilizerCode1.GKPRepititionNum + 1):
	StateDM = SystemDM.ptrace(i)
	PositionArray[i] = np.angle((StateDM*GKPStabilizerCode1.S_position).tr())/(np.sqrt(2*np.pi))
	MomentumArray[i] = np.angle((StateDM*GKPStabilizerCode1.S_momentum).tr())/(np.sqrt(2*np.pi))

# %%
GKPStabilizerCode1.S_position

# %%
print(np.angle((StateDM*GKPStabilizerCode1.S_position).tr())/(np.sqrt(2*np.pi)))


# %%
TestState = GKPStabilizerCode1.SystemState

TestRho = TestState*TestState.dag()

RhoSignal = TestRho.ptrace(0)
RhoAncilla = TestRho.ptrace(1)

print((Rho).tr())
print((RhoSignal*RhoSignal).tr())

