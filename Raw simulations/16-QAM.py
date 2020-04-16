# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:48:15 2020

@author: EIT6, Group 651
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt

# Parameter
L = 10000   # Length (Monte Carlo runs)
Es = 1      # Symbol energy
SNRdb = 20;  # Signal-noise radio db
k = 4;      # Number of bits (16-QAM -> 4 bits)

# Variables 
SNR = 10**(SNRdb/10)  # from db to linear 
M = 2**k              # Number of symbols
N0_half = Es/(SNR*2) # Noise variance (equation 2.37 and 2.38 in report)

# Noise
w = np.random.randn(2,L)*np.sqrt(N0_half)

# Sample seq. 
U = np.random.randint(0,2,[k,L]) # The number of bits to simulate is k*L

# Arrays
x = np.zeros([2,L])
U_hat = np.zeros([k,L])

### TRANSMITTER

# Signal Constellation (the possible symbol values)
s = np.zeros([2,M])


for i in range(M): 
    if i < 4:
        s[:,i] = np.array([(1/3)*np.sqrt(Es)*np.cos(2*np.pi*((i)/4)+np.pi/4), (1/3)*np.sqrt(Es)*np.sin(2*np.pi*((i)/4)+np.pi/4)])
    elif i == 4:
        s[:,i] = np.array([3*s[0,0],s[1,0]])
    elif i == 5:
        s[:,i] = np.array([s[0,0],3*s[1,0]])
    elif i == 6:
        s[:,i] = np.array([3*s[0,1],s[1,1]])
    elif i == 7:
        s[:,i] = np.array([s[0,1],3*s[1,1]])
    elif i == 8:
        s[:,i] = np.array([3*s[0,2],s[1,2]])
    elif i == 9:
        s[:,i] = np.array([s[0,2],3*s[1,2]])
    elif i == 10:
        s[:,i] = np.array([3*s[0,3],s[1,3]])
    elif i == 11:
        s[:,i] = np.array([s[0,3],3*s[1,3]])
    else:
        s[:,i] = np.array([np.sqrt(Es)*np.cos(2*np.pi*((i-12)/4)+np.pi/4), np.sqrt(Es)*np.sin(2*np.pi*((i-12)/4)+np.pi/4)])
# Side 356-359 Proakis and Salehi (which formulas?)
        
# Mapping LUT (maps a symbol to a signal)
for i in range(L):
    if np.array_equal(U[:,i],np.array([0,0,0,0])):
        x[:,i] = s[:,0]
    elif np.array_equal(U[:,i],np.array([0,0,0,1])):
        x[:,i] = s[:,1]
    elif np.array_equal(U[:,i],np.array([0,0,1,0])):
        x[:,i] = s[:,2]
    elif np.array_equal(U[:,i],np.array([0,0,1,1])):
        x[:,i] = s[:,3]
    elif np.array_equal(U[:,i],np.array([0,1,0,0])):
        x[:,i] = s[:,4]
    elif np.array_equal(U[:,i],np.array([0,1,0,1])):
        x[:,i] = s[:,5]
    elif np.array_equal(U[:,i],np.array([0,1,1,0])):
        x[:,i] = s[:,6]
    elif np.array_equal(U[:,i],np.array([0,1,1,1])):
        x[:,i] = s[:,7]
    elif np.array_equal(U[:,i],np.array([1,0,0,0])):
        x[:,i] = s[:,8]
    elif np.array_equal(U[:,i],np.array([1,0,0,1])):
        x[:,i] = s[:,9]
    elif np.array_equal(U[:,i],np.array([1,0,1,0])):
        x[:,i] = s[:,10]
    elif np.array_equal(U[:,i],np.array([1,0,1,1])):
        x[:,i] = s[:,11]
    elif np.array_equal(U[:,i],np.array([1,1,0,0])):
        x[:,i] = s[:,12]
    elif np.array_equal(U[:,i],np.array([1,1,0,1])):
        x[:,i] = s[:,13]
    elif np.array_equal(U[:,i],np.array([1,1,1,0])):
        x[:,i] = s[:,14]
    else:
        x[:,i] = s[:,15]

### CHANNEL        
        
# Adding noise
y = x + w

### RECEIVER

# Plot
plt.scatter(y[0,:],y[1,:])
plt.scatter(x[0,:],x[1,:])
plt.show()

# Decode / inv(encode), se side 91 lecture notes

d1 = np.sqrt(2)*(Es/6)  # Perpendicular distance from x-axis to the first point.
d2 = np.sqrt(2)*(Es/2)  # Perpendicular distance from x-axis to the last point.

for i in range(L):
    if y[0,i] < -d2+d1:
        if y[1,i] < -d2+d1:
            U_hat[:,i] = np.array([1,1,1,0])  # Checked
        elif y[1,i] >= -d2+d1 and y[1,i] < 0:
            U_hat[:,i] = np.array([1,0,0,0])  # Checked
        elif y[1,i] >= 0 and y[1,i] < d2-d1:
            U_hat[:,i] = np.array([0,1,1,0])  # Checked
        else:
            U_hat[:,i] = np.array([1,1,0,1])  # Checked
    elif y[0,i] >= -d2+d1 and y[0,i] < 0:
        if y[1,i] < -d2+d1:
            U_hat[:,i] = np.array([1,0,0,1])  # Checked
        elif y[1,i] >= -d2+d1 and y[1,i] < 0:
            U_hat[:,i] = np.array([0,0,1,0])  # Checked
        elif y[1,i] >= 0 and y[1,i] < d2-d1:
            U_hat[:,i] = np.array([0,0,0,1])  # Checked
        else:
            U_hat[:,i] = np.array([0,1,1,1])  # Checked
    elif y[0,i] >= 0 and y[0,i] < d2-d1:
        if y[1,i] < -d2+d1:
            U_hat[:,i] = np.array([1,0,1,1])  # Checked
        elif y[1,i] >= -d2+d1 and y[1,i] < 0:
            U_hat[:,i] = np.array([0,0,1,1])  # Checked
        elif y[1,i] >= 0 and y[1,i] < d2-d1:
            U_hat[:,i] = np.array([0,0,0,0])  # Checked
        else:
            U_hat[:,i] = np.array([0,1,0,1])  # Checked
    else:
        if y[1,i] < -d2+d1:
            U_hat[:,i] = np.array([1,1,1,1])  # Checked
        elif y[1,i] >= -d2+d1 and y[1,i] < 0:
            U_hat[:,i] = np.array([1,0,1,0])  # Checked
        elif y[1,i] >= 0 and y[1,i] < d2-d1:
            U_hat[:,i] = np.array([0,1,0,0])  # Checked
        else:
            U_hat[:,i] = np.array([1,1,0,0])  # Checked

### ERROR ANALYSIS

# Number of errors
N_error_bit = sum(sum(abs(U_hat-U)))

BEM = abs(U_hat - U)        # Bit error matrix
BCM = np.ones([4,L]) - BEM  # Bit correct matrix
SCM = BCM[0,:]*BCM[1,:]     # Symbol correct matrix

N_error_sym = L - sum(SCM)

pb = N_error_bit / (L*k)
ps = N_error_sym / L

print("BER: ", pb)
print("SER: ", ps)