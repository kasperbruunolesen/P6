# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:12:41 2020

@author: almsk
"""
import numpy as np
import matplotlib.pyplot as plt


### Parameter
L = 10000000   # Length (Monte Carlo runs)
Es = 1      # Symbol energy
SNRdb_bit = 20;  # Signal-noise radio db
k = 6      # Number of bits


# Variables 
SNR = (10**(SNRdb_bit/10))*k  # from db to linear 
M = 2**k              # Number of symbols (** = ^2)
N0_half = Es/(SNR*2) # Noise variance


# Sample seq. 
U = np.random.randint(0,2,[L,k])

# Arrays
x = np.zeros([L,2])
U_hat = np.zeros([k,L])



### Constellation 
s = np.zeros([2,M])
d = np.sqrt(Es/(42))
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
s[:,12] = np.array([5*d,d])
s[:,13] = np.array([d,5*d])
s[:,14] = np.array([-5*d,d])
s[:,15] = np.array([-d,5*d])
s[:,16] = np.array([-5*d,-d])
s[:,17] = np.array([-d,-5*d])
s[:,18] = np.array([5*d,-d])
s[:,19] = np.array([d,-5*d])
s[:,20] = np.array([7*d,d])
s[:,21] = np.array([d,7*d])
s[:,22] = np.array([-7*d,d])
s[:,23] = np.array([-d,7*d])
s[:,24] = np.array([-7*d,-d])
s[:,25] = np.array([-d,-7*d])
s[:,26] = np.array([7*d,-d])
s[:,27] = np.array([d,-7*d])
s[:,28] = np.array([3*d,3*d])
s[:,29] = np.array([-3*d,3*d])
s[:,30] = np.array([-3*d,-3*d])
s[:,31] = np.array([3*d,-3*d])
s[:,32] = np.array([5*d,3*d])
s[:,33] = np.array([3*d,5*d])
s[:,34] = np.array([-5*d,3*d])
s[:,35] = np.array([-3*d,5*d])
s[:,36] = np.array([-5*d,-3*d])
s[:,37] = np.array([-3*d,-5*d])
s[:,38] = np.array([5*d,-3*d])
s[:,39] = np.array([3*d,-5*d])
s[:,40] = np.array([7*d,3*d])
s[:,41] = np.array([3*d,7*d])
s[:,42] = np.array([-7*d,3*d])
s[:,43] = np.array([-3*d,7*d])
s[:,44] = np.array([-7*d,-3*d])
s[:,45] = np.array([-3*d,-7*d])
s[:,46] = np.array([7*d,-3*d])
s[:,47] = np.array([3*d,-7*d])
s[:,48] = np.array([5*d,5*d])
s[:,49] = np.array([-5*d,5*d])
s[:,50] = np.array([-5*d,-5*d])
s[:,51] = np.array([5*d,-5*d])
s[:,52] = np.array([7*d,5*d])
s[:,53] = np.array([5*d,7*d])
s[:,54] = np.array([-7*d,5*d])
s[:,55] = np.array([-5*d,7*d])
s[:,56] = np.array([-7*d,-5*d])
s[:,57] = np.array([-5*d,-7*d])
s[:,58] = np.array([7*d,-5*d])
s[:,59] = np.array([5*d,-7*d])
s[:,60] = np.array([7*d,7*d])
s[:,61] = np.array([-7*d,7*d])
s[:,62] = np.array([-7*d,-7*d])
s[:,63] = np.array([7*d,-7*d])

# Constellation Sorted
s_sort = np.zeros_like(s)
s_sort[:,0] = s[:,62]  # '000000' -> s62
s_sort[:,1] = s[:,56]  # '000001' -> s56
s_sort[:,2] = s[:,24]  # '000010' -> s24
s_sort[:,3] = s[:,44]  # '000011' -> s44
s_sort[:,4] = s[:,61]  # '000100' -> s61
s_sort[:,5] = s[:,54]  # '000101' -> s54
s_sort[:,6] = s[:,22]  # '000110' -> s22
s_sort[:,7] = s[:,42]  # '000111' -> s42
s_sort[:,8] = s[:,57]  # '001000' -> s57
s_sort[:,9] = s[:,50]  # '001001' -> s50
s_sort[:,10] = s[:,16] # '001010' -> s16
s_sort[:,11] = s[:,36] # '001011' -> s36
s_sort[:,12] = s[:,55] # '001100' -> s55
s_sort[:,13] = s[:,49] # '001101' -> s49
s_sort[:,14] = s[:,14] # '001110' -> s14
s_sort[:,15] = s[:,34] # '001111' -> s34
s_sort[:,16] = s[:,25] # '010000' -> s25
s_sort[:,17] = s[:,17] # '010001' -> s17
s_sort[:,18] = s[:,2]  # '010010' -> s2
s_sort[:,19] = s[:,9]  # '010011' -> s9
s_sort[:,20] = s[:,23] # '010100' -> s23
s_sort[:,21] = s[:,15] # '010101' -> s15
s_sort[:,22] = s[:,1]  # '010110' -> s1
s_sort[:,23] = s[:,7]  # '010111' -> s7
s_sort[:,24] = s[:,45] # '011000' -> s45
s_sort[:,25] = s[:,37] # '011001' -> s37
s_sort[:,26] = s[:,8]  # '011010' -> s8
s_sort[:,27] = s[:,30] # '011011' -> s30
s_sort[:,28] = s[:,43] # '011100' -> s43
s_sort[:,29] = s[:,35] # '011101' -> s35
s_sort[:,30] = s[:,6]  # '011110' -> s6
s_sort[:,31] = s[:,29] # '011111' -> s29
s_sort[:,32] = s[:,63] # '100000' -> s63
s_sort[:,33] = s[:,58] # '100001' -> s58
s_sort[:,34] = s[:,26] # '100010' -> s26
s_sort[:,35] = s[:,46] # '100011' -> s46
s_sort[:,36] = s[:,60] # '100100' -> s60
s_sort[:,37] = s[:,52] # '100101' -> s52
s_sort[:,38] = s[:,20] # '100110' -> s20
s_sort[:,39] = s[:,40] # '100111' -> s40
s_sort[:,40] = s[:,59] # '101000' -> s59
s_sort[:,41] = s[:,51] # '101001' -> s51
s_sort[:,42] = s[:,18] # '101010' -> s18
s_sort[:,43] = s[:,38] # '101011' -> s38
s_sort[:,44] = s[:,53] # '101100' -> s53
s_sort[:,45] = s[:,48] # '101101' -> s48
s_sort[:,46] = s[:,12] # '101110' -> s12
s_sort[:,47] = s[:,32] # '101111' -> s32
s_sort[:,48] = s[:,27] # '110000' -> s27
s_sort[:,49] = s[:,19] # '110001' -> s19
s_sort[:,50] = s[:,3]  # '110010' -> s3
s_sort[:,51] = s[:,11] # '110011' -> s11
s_sort[:,52] = s[:,21] # '110100' -> s21
s_sort[:,53] = s[:,13] # '110101' -> s13
s_sort[:,54] = s[:,0]  # '110110' -> s0
s_sort[:,55] = s[:,5]  # '110111' -> s5
s_sort[:,56] = s[:,47] # '111000' -> s47
s_sort[:,57] = s[:,39] # '111001' -> s39
s_sort[:,58] = s[:,10] # '111010' -> s10
s_sort[:,59] = s[:,31] # '111011' -> s31
s_sort[:,60] = s[:,41] # '111100' -> s41
s_sort[:,61] = s[:,33] # '111101' -> s33
s_sort[:,62] = s[:,4]  # '111110' -> s4
s_sort[:,63] = s[:,28] # '111111' -> s28


# Check mean symbol power:
total_power = (s[:,:]**2).sum()
mean_power = total_power/M
print("The mean symbol power is", mean_power)   

### Base symbols
b = np.zeros([M,k])
for i5 in range(2):
    for i4 in range(2):
        for i3 in range(2):
            for i2 in range(2):
                for i1 in range(2):
                    for i0 in range(2):
                        b[32*i5 + 16*i4 + 8*i3 + 4*i2 + 2*i1 + i0,:] = [i5,i4,i3,i2,i1,i0]


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

