# %%
import sys
sys.path.append('/Users/qian/Documents/Research/JiangGroupProjects/QEC_with_bosinic_code/codes')

import numpy as np 
from qutip import *
import matplotlib.pyplot as plt 
from CreateGKPState import *
from PlotWignerDM import *
from PlotWignerState import *
import copy


class GKPStabilizerCode(object):
	'''

	'''


	# Defaults
	OscillatorDim = 20	# Dimension of each oscillator mode, assume all modes have the same dimension for now
	GKPSumNum = 4	# Summation number of each GKP state
	GKPRepititionNum = 2
	EncodingOperator = None 	# The encoding operator which primarily defines a GKP stabilizer code
	DecodingOperator = None
	SystemState = None 	# The whole system state including data mode and ancilla modes 
	AncillaStateList = []


	# Initizlization
	def __init__(self, EncodingOperator_input = None, DecodingOperator_input = None, OscillatorDim_input = 40, GKPRepititionNum_input = 2, GKPSumNum_input = 4):
		self.EncodingOperator = copy.deepcopy(EncodingOperator_input)
		self.DecodingOperator = copy.deepcopy(DecodingOperator_input)
		self.OscillatorDim = copy.deepcopy(OscillatorDim_input)
		self.GKPSumNum = copy.deepcopy(GKPSumNum_input)
		self.GKPRepititionNum = copy.deepcopy(GKPRepititionNum_input)
		self.AncillaStateList = [CreateGKPState(self.OscillatorDim, self.GKPSumNum) for i in range(self.GKPRepititionNum)]
		self.S_position = (1j*np.sqrt(2*np.pi)*position(self.OscillatorDim)).expm()
		self.S_momentum = (- 1j*np.sqrt(2*np.pi)*momentum(self.OscillatorDim)).expm()

	#	Input the initial state of the data mode
	def InputDataMode(self, DataModeState_input):
		#	The initial system state is the tensor product of the Data mode and the ancilla modes
		self.SystemState = tensor([DataModeState_input] + self.AncillaStateList)

	#	Define the encoding process
	def Encode(self):
		self.SystemState = self.EncodingOperator*self.SystemState

	#	Define the process when the encoded system goes through a noisy channel
	def ThroughNoisyChannel(self, NoiseOperList):
		TotalNoiseOper = 1
		for NoiseOper in NoiseOperList:
			TotalNoiseOper *= NoiseOper
		self.SystemState = TotalNoiseOper*self.SystemState

	#	Define the decoding process
	def Decode(self):
		self.SystemState = self.DecodingOperator*self.SystemState

	#	
	def Measure(self):
		PositionArray = np.zeros(self.GKPRepititionNum + 1)
		MomentumArray = np.zeros(self.GKPRepititionNum + 1)
		# DataModeDM = SystemState.ptrace(0)
		# PositionArray[0] = (DataModeDM*S_position).tr()
		# MomentumArray[0] = (DataModeDM*S_momentum).tr()
		SystemDM = self.SystemState*self.SystemState.dag()
		for i in range(self.GKPRepititionNum + 1):
			StateDM = SystemDM.ptrace(i)
			PositionArray[i] = np.angle((StateDM*self.S_position).tr())/(np.sqrt(2*np.pi))
			MomentumArray[i] = np.angle((StateDM*self.S_momentum).tr())/(np.sqrt(2*np.pi))
		return [PositionArray, MomentumArray]

		


# %%

