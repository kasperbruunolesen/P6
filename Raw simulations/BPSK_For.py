# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:33:12 2020

@author: EIT6, Group 651
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt
import time
startTime = time.time()

# Parameter
L = 10000       # Length (Monte Carlo runs)
Es = 1          # Symbol energy (technically not needed, as long as working with SNR)
SNRdb_bit = 3   # Signal-noise radio db
k = 1;          # Number of bits
N = 100         # Number of loops


# Variables 
SNR = (10**(SNRdb_bit/10))*k     # from db to linear
M = 2**k                         # Number of symbols
N0_half = Es/(SNR*2)             # Noise variance (eq 2.38 in report) (N0_half = sigma**2)
N_error_bit = 0                  # Number of error bits declaration
N_error_sym = 0                  # Number of error symbols declaration


### TRANSMITTER BEGIN ###

# Signal Constellation
s = np.zeros([1,M]) # For the basis functions
s[0,0]=1
s[0,1]=-1

for n in range(N):
    # Sample seq. 
    U = np.random.randint(0,2,[1,L]) # Making 2*L bits (random bit stream)
    # Arrays
    x = np.zeros([2,L]) # The sent signal
    U_hat = np.zeros([1,L]) # The decoded input to receiver    
        
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
    
# =============================================================================
#     # Plot
#     plt.scatter(y[0,:],y[1,:])
#     plt.scatter(x[0,:],x[1,:])
#     plt.show()
# =============================================================================
    
    
    
    for i in range(L):
        if y[0,i] > 0:
                U_hat[:,i] = np.array([1])
        else:
                U_hat[:,i] = np.array([0])
    
    ### RECEIVER END ###
    
    ### ERROR ANALYSIS BEGIN ###
    
    # Number of errors
    
    
    BEM = abs(U_hat - U)        # Bit error matrix
    N_error_bit = sum(sum(BEM)) + N_error_bit
    
    BCM = np.ones([2,L]) - BEM  # Bit correct matrix
    SCM = np.ones([1,L])        # Symbol correct matrix
    for i in range(k):
        SCM = BCM[i,:] * SCM
    
    N_error_sym = L - sum(SCM) + N_error_sym
    print("Run: ", n+1 , " of ", N)

pb = N_error_bit / ((L*N)*k)
ps = N_error_sym / (L*N)

print("Number of symbols", L*N)
print("Number of bit errors", N_error_bit)

print("BER: ", pb)
print("SER: ", ps)
print("Execution time was", time.time()-startTime, "seconds")     
     

### ERROR ANALYSIS END ### 