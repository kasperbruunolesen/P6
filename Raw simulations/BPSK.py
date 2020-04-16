# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:33:12 2020

@author: EIT6, Group 651
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt

# Parameter
L = 10000   # Length (Monte Carlo runs)
Es = 1      # Symbol energy (technically not needed, as long as working with SNR)
SNRdb = 3;  # Signal-noise radio db
k = 1;      # Number of bits

# Variables 
SNR = 10**(SNRdb/10)  # from db to linear
M = 2**k              # Number of symbols
N0_half = Es/(SNR*2) # Noise variance (eq 2.38 in report) (N0_half = sigma**2)

# Sample seq. 
U = np.random.randint(0,2,[1,L]) # Making 2*L bits (random bit stream)
# Arrays
x = np.zeros([2,L]) # The sent signal
U_hat = np.zeros([1,L]) # The decoded input to receiver

### TRANSMITTER BEGIN ###

# Signal Constellation
s = np.zeros([1,M]) # For the basis functions
s[0,0]=1
s[0,1]=-1
    
    
# Mapping LUT. Maps the bit stream to the symbols for transmitting
for i in range(L):
    if np.array_equal(U[:,i],np.array([1])):
        x[0,i] = s[:,0]
    else:
        x[0,i] = s[:,1]

### TRANSMITTER END ###

### CHANNEL BEGIN ###

# Noise array
w = np.random.randn(2,L)*np.sqrt(N0_half)        
# Adding noise
y = x + w

### CHANNEL END ###

### RECEIVER BEGIN ###

# Plot
plt.scatter(y[0,:],y[1,:])
plt.scatter(x[0,:],x[1,:])
plt.show()



for i in range(L):
    if y[0,i] > 0:
            U_hat[:,i] = np.array([1])
    else:
            U_hat[:,i] = np.array([0])

### RECEIVER END ###

### ERROR ANALYSIS BEGIN ###

# Number of errors
N_error_bit = sum(sum(abs(U_hat-U)))

BEM = abs(U_hat - U)        # Bit error matrix
BCM = np.ones([2,L]) - BEM  # Bit correct matrix
SCM = BCM[0,:]*BCM[1,:]     # Symbol correct matrix

N_error_sym = L - sum(SCM)

pb = N_error_bit / (L*2)
ps = N_error_sym / L

print("BER: ", pb)
print("SER: ", ps)       

### ERROR ANALYSIS END ### 