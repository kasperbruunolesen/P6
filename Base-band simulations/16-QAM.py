# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:48:15 2020

@author: EIT6, Group 651
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt

### Parameter
L = 10000   # Length (Monte Carlo runs)
Es = 1      # Symbol energy
SNRdb = 10;  # Signal-noise radio db (per bit)
k = 4;      # Number of bits (16-QAM -> 4 bits)

# Variables 
SNR = (10**(SNRdb/10))*k  # from db to linear and to snr/symbol (important!)
M = 2**k              # Number of symbols
N0_half = Es/(SNR*2) # Noise variance (equation 2.37 and 2.38 in report)

# Sample seq. 
U = np.random.randint(0,2,[L,k])

# Arrays
x = np.zeros([L,2])
U_hat = np.zeros([k,L])


### Constellation
s = np.zeros([2,M])
d = np.sqrt(Es/10)
s[:,0] = np.array([d,d])
s[:,1] = np.array([-d,d])
s[:,2] = np.array([-d,-d])
s[:,3] = np.array([d,-d])
s[:,4] = np.array([3*d,d])
s[:,5] = np.array([d,3*d])
s[:,6] = np.array([-3*d,d])
s[:,7] = np.array([-d,3*d])
s[:,8] = np.array([-3*d,-d])
s[:,9] = np.array([-d,-3*d])
s[:,10] = np.array([3*d,-d])
s[:,11] = np.array([d,-3*d])
s[:,12] = np.array([3*d,3*d])
s[:,13] = np.array([-3*d,3*d])
s[:,14] = np.array([-3*d,-3*d])
s[:,15] = np.array([3*d,-3*d])



# Constellation Sorted
s_sort = np.zeros_like(s)
s_sort[:,0] = s[:,14]  # '0000' -> s14
s_sort[:,1] = s[:,8]   # '0001' -> s8
s_sort[:,2] = s[:,13]  # '0010' -> s13
s_sort[:,3] = s[:,6]   # '0011' -> s6
s_sort[:,4] = s[:,9]   # '0100' -> s9
s_sort[:,5] = s[:,2]   # '0101' -> s2
s_sort[:,6] = s[:,7]   # '0110' -> s7
s_sort[:,7] = s[:,1]   # '0111' -> s1
s_sort[:,8] = s[:,15]  # '1000' -> s15
s_sort[:,9] = s[:,10]  # '1001' -> s10
s_sort[:,10] = s[:,12] # '1010' -> s12
s_sort[:,11] = s[:,4]  # '1011' -> s4
s_sort[:,12] = s[:,11] # '1100' -> s11
s_sort[:,13] = s[:,3]  # '1101' -> s3
s_sort[:,14] = s[:,5]  # '1110' -> s5
s_sort[:,15] = s[:,0]  # '1111' -> s0


# Check mean symbol power:
total_power = (s[:,:]**2).sum()
mean_power = total_power/M
print("The mean symbol power is", mean_power)   

### Base symbols
b = np.zeros([M,k])
for i3 in range(2):
    for i2 in range(2):
        for i1 in range(2):
            for i0 in range(2):
                b[8*i3 + 4*i2 + 2*i1 + i0,:] = [i3,i2,i1,i0]


### Mapping LUT

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



#Adding Noise
y = x + w

# Plot
plt.scatter(y[:,0],y[:,1],label = 'Data with noise')
plt.scatter(x[:,0],x[:,1],label = 'Data without noise')
plt.title(f"16-QAM, SNR: {SNRdb}, No. of symbols: {L}")
plt.xlabel('I', color='#1C2833')
plt.ylabel('Q', color='#1C2833')
plt.legend(loc='upper right')
plt.show()


### Decode / inv(encode)
# se side 91 lecture notes

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