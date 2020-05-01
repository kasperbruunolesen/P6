# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:33:12 2020

@author: EIT6, Group 651
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt

### Parameter
L = 10000           # Length (Monte Carlo runs)
Es = 1              # Symbol energy
SNRdb_bit = 1;      # Signal-noise radio db
k = 2;              # Number of bits

# Variables 
SNR = (10**(SNRdb_bit/10))*k  # from db to linear
M = 2**k              # Number of symbols
N0_half = Es/(SNR*2) # Noise variance (eq 2.38 in report) (N0_half = sigma**2)


# Sample seq. 
U = np.random.randint(0,2,[L,k]) # Making L*k bits (random bit stream)

# Arrays
x = np.zeros([L,2])              # The sent signal
U_hat = np.zeros([k,L])          # The decoded input to receiver


### Constellation
s = np.zeros([2,M]) # For the basis functions
for i in range(M): 
    s[:,i] = np.array([np.sqrt(Es)*np.cos(2*np.pi*((i)/(M))), np.sqrt(Es)*np.sin(2*np.pi*((i)/(M)))]) 
# page 356 in Proakkis and Salehi. The optimal splitting of M-PSK is given here.
  
# Constellation Sorted
s_sort = np.zeros_like(s)
s_sort[:,0] = s[:,0]  # '00' -> s0
s_sort[:,1] = s[:,1]  # '01' -> s1
s_sort[:,2] = s[:,3]  # '10' -> s3
s_sort[:,3] = s[:,2]  # '11' -> s2    
    
# Check mean symbol power:
total_power = (s[:,:]**2).sum()
mean_power = total_power/M
print("The mean symbol power is", mean_power)       

### Base symbols
b = np.zeros([M,k])
for i1 in range(2):
    for i0 in range(2):
        b[2*i1 + i0,:] = [i1,i0]


### Mapping LUT
# Maps the bit stream to the symbols for transmitting

# Checks if bit combination if it is in U and return a array with the places index number. 
# The index number is used to replace the given places in x with the correlated symbol
for i in range(M):
    x[np.argwhere(np.all(U == b[i,:], axis=1))] = s_sort[:,i]  


#Calculate mean symbol power (validation)
total_power_generated = (x[:,:]**2).sum()
mean_power_generated = total_power_generated/L
print("The mean generated symbol power is", mean_power_generated) 


### Channel

# Noise
w = np.random.randn(L,2)*np.sqrt(N0_half)

#Calculate mean noise power (validation)
total_noise_generated = (w[:,:]**2).sum()
mean_noise_generated = total_noise_generated/L
print("The mean generated noise power is", mean_noise_generated)
print("The expected noise power is", N0_half*2) 
      
# Adding noise
y = x + w

### Decode / inv(encode)
# see page 91 lecture notes

# Creates a matrix which has the distance from the signal to every symbol
Dist = np.zeros([L,M])
for i in range(M):
    Dist[:,i] = (np.sqrt((s_sort[0,i]-y[:,0])**2 + (s_sort[1,i] - y[:,1])**2))

# return the index for the places each min occour in each row and uses them in 'b'
# to put the correct symbol inside U_hat
U_hat = b[Dist.argmin(axis = 1),:] 


### ERROR ANALYSIS 

# Number of errors
BEM = abs(U_hat - U)        # Bit error matrix
BCM = np.ones([L,k]) - BEM  # Bit correct matrix
SCM = np.ones([1,L])        # Symbol correct vector
for i in range(k):
    SCM = SCM*BCM[:,i] #If it reaches a 0 down the road, the SCM index becomes a zero

N_error_bit = sum(sum(abs(BEM)))

N_error_sym = L - sum(sum(SCM))

pb = N_error_bit / (L*k)
ps = N_error_sym / L

print("Number of symbols", L)
print("Number of bits", L*k)
print("Number of symbol errors", N_error_sym)
print("Number of bit errors", N_error_bit)
print("BER: ", pb)
print("SER: ", ps)      
