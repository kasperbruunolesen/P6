# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:48:20 2020

@author: EIT6, Group 651
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt

### Parameter
L = 10000   # Length (Monte Carlo runs)
Es = 1      # Symbol energy
SNRdb = 20;  # Signal-noise radio db
k = 6;      # Number of bits

# Variables 
SNR = 10**(SNRdb/10)  # from db to lineary 
M = 2**k              # Number of symbols
N0_half = Es/(SNR*2) # Noise variance

# Noise
w = np.random.randn(2,L)*np.sqrt(N0_half)

# Sample seq. 
U = np.random.randint(0,2,[k,L])
#U = np.array([[1,1],[1,1],[1,1],[1,1],[0,1],[0,1]])

# Arrays
x = np.zeros([2,L])
U_hat = np.zeros([k,L])

### Signal Constellation
s = np.zeros([2,M])


for i in range(M): 
    if i < 4:
        s[:,i] = np.array([(1/7)*np.sqrt(Es)*np.cos(2*np.pi*((i)/4)+np.pi/4), (1/7)*np.sqrt(Es)*np.sin(2*np.pi*((i)/4)+np.pi/4)])
    elif i >= 4 and i < 12:
        if i == 4:
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
        else: # 11
            s[:,i] = np.array([s[0,3],3*s[1,3]])
    elif i >= 12 and i < 20:
        if i == 12:
            s[:,i] = np.array([5*s[0,0],s[1,0]])
        elif i == 13:
            s[:,i] = np.array([s[0,0],5*s[1,0]])
        elif i == 14:
            s[:,i] = np.array([5*s[0,1],s[1,1]])
        elif i == 15:
            s[:,i] = np.array([s[0,1],5*s[1,1]])
        elif i == 16:
            s[:,i] = np.array([5*s[0,2],s[1,2]])
        elif i == 17:
            s[:,i] = np.array([s[0,2],5*s[1,2]])
        elif i == 18:
            s[:,i] = np.array([5*s[0,3],s[1,3]])
        else: #  19
            s[:,i] = np.array([s[0,3],5*s[1,3]])
    elif i >= 20 and i < 28:
        if i == 20:
            s[:,i] = np.array([7*s[0,0],s[1,0]])
        elif i == 21:
            s[:,i] = np.array([s[0,0],7*s[1,0]])
        elif i == 22:
            s[:,i] = np.array([7*s[0,1],s[1,1]])
        elif i == 23:
            s[:,i] = np.array([s[0,1],7*s[1,1]])
        elif i == 24:
            s[:,i] = np.array([7*s[0,2],s[1,2]])
        elif i == 25:
            s[:,i] = np.array([s[0,2],7*s[1,2]])
        elif i == 26:
            s[:,i] = np.array([7*s[0,3],s[1,3]])
        else: #  27
            s[:,i] = np.array([s[0,3],7*s[1,3]])
    if i >= 28 and i < 32:
        s[:,i] = np.array([(3/7)*np.sqrt(Es)*np.cos(2*np.pi*((i-28)/4)+np.pi/4), (3/7)*np.sqrt(Es)*np.sin(2*np.pi*((i-28)/4)+np.pi/4)])
    elif i >= 32 and i < 40:
        if i == 32:
            s[:,i] = np.array([(5/3)*s[0,28],s[1,28]])
        elif i == 33:
            s[:,i] = np.array([s[0,28],(5/3)*s[1,28]])
        elif i == 34:
            s[:,i] = np.array([(5/3)*s[0,29],s[1,29]])
        elif i == 35:
            s[:,i] = np.array([s[0,29],(5/3)*s[1,29]])
        elif i == 36:
            s[:,i] = np.array([(5/3)*s[0,30],s[1,30]])
        elif i == 37:
            s[:,i] = np.array([s[0,30],(5/3)*s[1,30]])
        elif i == 38:
            s[:,i] = np.array([(5/3)*s[0,31],s[1,31]])
        else: #  39
            s[:,i] = np.array([s[0,31],(5/3)*s[1,31]])
    elif i >= 40 and i < 48:
        if i == 40:
            s[:,i] = np.array([(7/3)*s[0,28],s[1,28]])
        elif i == 41:
            s[:,i] = np.array([s[0,28],(7/3)*s[1,28]])
        elif i == 42:
            s[:,i] = np.array([(7/3)*s[0,29],s[1,29]])
        elif i == 43:
            s[:,i] = np.array([s[0,29],(7/3)*s[1,29]])
        elif i == 44:
            s[:,i] = np.array([(7/3)*s[0,30],s[1,30]])
        elif i == 45:
            s[:,i] = np.array([s[0,30],(7/3)*s[1,30]])
        elif i == 46:
            s[:,i] = np.array([(7/3)*s[0,31],s[1,31]])
        else: #  47
            s[:,i] = np.array([s[0,31],(7/3)*s[1,31]])
    if i >= 48 and i < 52:
        s[:,i] = np.array([(5/7)*np.sqrt(Es)*np.cos(2*np.pi*((i-48)/4)+np.pi/4), (5/7)*np.sqrt(Es)*np.sin(2*np.pi*((i-48)/4)+np.pi/4)])
    elif i >= 52 and i < 60:
        if i == 52:
            s[:,i] = np.array([(7/5)*s[0,48],s[1,48]])
        elif i == 53:
            s[:,i] = np.array([s[0,48],(7/5)*s[1,48]])
        elif i == 54:
            s[:,i] = np.array([(7/5)*s[0,49],s[1,49]])
        elif i == 55:
            s[:,i] = np.array([s[0,49],(7/5)*s[1,49]])
        elif i == 56:
            s[:,i] = np.array([(7/5)*s[0,50],s[1,50]])
        elif i == 57:
            s[:,i] = np.array([s[0,50],(7/5)*s[1,50]])
        elif i == 58:
            s[:,i] = np.array([(7/5)*s[0,51],s[1,51]])
        else: #  59
            s[:,i] = np.array([s[0,51],(7/5)*s[1,51]]) 
    if i >= 60 and i < 64:
        s[:,i] = np.array([(7/7)*np.sqrt(Es)*np.cos(2*np.pi*((i-60)/4)+np.pi/4), (7/7)*np.sqrt(Es)*np.sin(2*np.pi*((i-60)/4)+np.pi/4)])  
# Side 356-359 Proakkis and Salehi
        
### Mapping LUT
for i in range(L):
    if U[0,i] == 0:
        if U[1,i] == 0:
            if U[2,i] == 0:
                if np.array_equal(U[:,i],np.array([0,0,0,0,0,0])):
                    x[:,i] = s[:,62]
                elif np.array_equal(U[:,i],np.array([0,0,0,0,0,1])):
                    x[:,i] = s[:,56]
                elif np.array_equal(U[:,i],np.array([0,0,0,0,1,0])):
                    x[:,i] = s[:,24]
                elif np.array_equal(U[:,i],np.array([0,0,0,0,1,1])):
                    x[:,i] = s[:,44]   
                elif np.array_equal(U[:,i],np.array([0,0,0,1,0,0])):
                    x[:,i] = s[:,61] 
                elif np.array_equal(U[:,i],np.array([0,0,0,1,0,1])):
                    x[:,i] = s[:,54]
                elif np.array_equal(U[:,i],np.array([0,0,0,1,1,0])):
                    x[:,i] = s[:,22] 
                else:
                    x[:,i] = s[:,42] 
            else:
                if np.array_equal(U[:,i],np.array([0,0,1,0,0,0])):
                    x[:,i] = s[:,57]
                elif np.array_equal(U[:,i],np.array([0,0,1,0,0,1])):
                    x[:,i] = s[:,50]
                elif np.array_equal(U[:,i],np.array([0,0,1,0,1,0])):
                    x[:,i] = s[:,16]
                elif np.array_equal(U[:,i],np.array([0,0,1,0,1,1])):
                    x[:,i] = s[:,36]   
                elif np.array_equal(U[:,i],np.array([0,0,1,1,0,0])):
                    x[:,i] = s[:,55] 
                elif np.array_equal(U[:,i],np.array([0,0,1,1,0,1])):
                    x[:,i] = s[:,49]
                elif np.array_equal(U[:,i],np.array([0,0,1,1,1,0])):
                    x[:,i] = s[:,14] 
                else:
                    x[:,i] = s[:,34]             
        else:
            if U[2,i] == 0:
                if np.array_equal(U[:,i],np.array([0,1,0,0,0,0])):
                    x[:,i] = s[:,25]
                elif np.array_equal(U[:,i],np.array([0,1,0,0,0,1])):
                    x[:,i] = s[:,17]
                elif np.array_equal(U[:,i],np.array([0,1,0,0,1,0])):
                    x[:,i] = s[:,2]
                elif np.array_equal(U[:,i],np.array([0,1,0,0,1,1])):
                    x[:,i] = s[:,9]   
                elif np.array_equal(U[:,i],np.array([0,1,0,1,0,0])):
                    x[:,i] = s[:,23] 
                elif np.array_equal(U[:,i],np.array([0,1,0,1,0,1])):
                    x[:,i] = s[:,15]
                elif np.array_equal(U[:,i],np.array([0,1,0,1,1,0])):
                    x[:,i] = s[:,1] 
                else:
                    x[:,i] = s[:,7] 
            else:
                if np.array_equal(U[:,i],np.array([0,1,1,0,0,0])):
                    x[:,i] = s[:,45]
                elif np.array_equal(U[:,i],np.array([0,1,1,0,0,1])):
                    x[:,i] = s[:,37]
                elif np.array_equal(U[:,i],np.array([0,1,1,0,1,0])):
                    x[:,i] = s[:,8]
                elif np.array_equal(U[:,i],np.array([0,1,1,0,1,1])):
                    x[:,i] = s[:,30]   
                elif np.array_equal(U[:,i],np.array([0,1,1,1,0,0])):
                    x[:,i] = s[:,43] 
                elif np.array_equal(U[:,i],np.array([0,1,1,1,0,1])):
                    x[:,i] = s[:,35]
                elif np.array_equal(U[:,i],np.array([0,1,1,1,1,0])):
                    x[:,i] = s[:,6] 
                else:
                    x[:,i] = s[:,29]           
    else:
        if U[1,i] == 0:
            if U[2,i] == 0:
                if np.array_equal(U[:,i],np.array([1,0,0,0,0,0])):
                    x[:,i] = s[:,63]
                elif np.array_equal(U[:,i],np.array([1,0,0,0,0,1])):
                    x[:,i] = s[:,58]
                elif np.array_equal(U[:,i],np.array([1,0,0,0,1,0])):
                    x[:,i] = s[:,26]
                elif np.array_equal(U[:,i],np.array([1,0,0,0,1,1])):
                    x[:,i] = s[:,46]   
                elif np.array_equal(U[:,i],np.array([1,0,0,1,0,0])):
                    x[:,i] = s[:,60] 
                elif np.array_equal(U[:,i],np.array([1,0,0,1,0,1])):
                    x[:,i] = s[:,52]
                elif np.array_equal(U[:,i],np.array([1,0,0,1,1,0])):
                    x[:,i] = s[:,20] 
                else:
                    x[:,i] = s[:,40] 
            else:
                if np.array_equal(U[:,i],np.array([1,0,1,0,0,0])):
                    x[:,i] = s[:,59]
                elif np.array_equal(U[:,i],np.array([1,0,1,0,0,1])):
                    x[:,i] = s[:,51]
                elif np.array_equal(U[:,i],np.array([1,0,1,0,1,0])):
                    x[:,i] = s[:,18]
                elif np.array_equal(U[:,i],np.array([1,0,1,0,1,1])):
                    x[:,i] = s[:,38]   
                elif np.array_equal(U[:,i],np.array([1,0,1,1,0,0])):
                    x[:,i] = s[:,53] 
                elif np.array_equal(U[:,i],np.array([1,0,1,1,0,1])):
                    x[:,i] = s[:,48]
                elif np.array_equal(U[:,i],np.array([1,0,1,1,1,0])):
                    x[:,i] = s[:,12] 
                else:
                    x[:,i] = s[:,32]             
        else:
            if U[2,i] == 0:
                if np.array_equal(U[:,i],np.array([1,1,0,0,0,0])):
                    x[:,i] = s[:,27]
                elif np.array_equal(U[:,i],np.array([1,1,0,0,0,1])):
                    x[:,i] = s[:,19]
                elif np.array_equal(U[:,i],np.array([1,1,0,0,1,0])):
                    x[:,i] = s[:,3]
                elif np.array_equal(U[:,i],np.array([1,1,0,0,1,1])):
                    x[:,i] = s[:,11]   
                elif np.array_equal(U[:,i],np.array([1,1,0,1,0,0])):
                    x[:,i] = s[:,21] 
                elif np.array_equal(U[:,i],np.array([1,1,0,1,0,1])):
                    x[:,i] = s[:,13]
                elif np.array_equal(U[:,i],np.array([1,1,0,1,1,0])):
                    x[:,i] = s[:,0] 
                else:
                    x[:,i] = s[:,5] 
            else:
                if np.array_equal(U[:,i],np.array([1,1,1,0,0,0])):
                    x[:,i] = s[:,47]
                elif np.array_equal(U[:,i],np.array([1,1,1,0,0,1])):
                    x[:,i] = s[:,39]
                elif np.array_equal(U[:,i],np.array([1,1,1,0,1,0])):
                    x[:,i] = s[:,10]
                elif np.array_equal(U[:,i],np.array([1,1,1,0,1,1])):
                    x[:,i] = s[:,31]   
                elif np.array_equal(U[:,i],np.array([1,1,1,1,0,0])):
                    x[:,i] = s[:,41] 
                elif np.array_equal(U[:,i],np.array([1,1,1,1,0,1])):
                    x[:,i] = s[:,33]
                elif np.array_equal(U[:,i],np.array([1,1,1,1,1,0])):
                    x[:,i] = s[:,4] 
                else:
                    x[:,i] = s[:,28]   

        
# Adding noise
y = x + w

# Plot
plt.scatter(y[0,:],y[1,:])
plt.scatter(x[0,:],x[1,:])
plt.show()

### Decode / inv(encode),
# see page 91 lecture notes

d1 = np.sqrt(2)*(Es/14)  # Afstand fra x-akse til første punkt. Vinkelret
d2 = np.sqrt(2)*(Es/2)  # Afstand fra x-akse til sidste punkt. Vinkelret

for i in range(L):
    if y[0,i] < 0:                                    # Venstre halvplan
        if y[0,i] < -6*d1:                            # Firkant 1
            if y[1,i] < -6*d1:                        # Punkt 62
                U_hat[:,i] = np.array([0,0,0,0,0,0])
            elif y[1,i] >= -6*d1 and y[1,i] < -4*d1:  # Punkt 56
                U_hat[:,i] = np.array([0,0,0,0,0,1])
            elif y[1,i] >= -4*d1 and y[1,i] < -2*d1:  # Punkt 44
                U_hat[:,i] = np.array([0,0,0,0,1,1])
            elif y[1,i] >= -2*d1 and y[1,i] < 0:      # Punkt 24
                U_hat[:,i] = np.array([0,0,0,0,1,0])
            elif y[1,i] >= 0 and y[1,i] < 2*d1:       # Punkt 22
                U_hat[:,i] = np.array([0,0,0,1,1,0])
            elif y[1,i] >= 2*d1 and y[1,i] < 4*d1:    # Punkt 42
                U_hat[:,i] = np.array([0,0,0,1,1,1])
            elif y[1,i] >= 4*d1 and y[1,i] < 6*d1:    # Punkt 54
                U_hat[:,i] = np.array([0,0,0,1,0,1])
            else:                                     # Punkt 61
                U_hat[:,i] = np.array([0,0,0,1,0,0])
        elif y[0,i] >= -6*d1 and y[0,i] < -4*d1:        # Firkant 2
            if y[1,i] < -6*d1:                        # Punkt 57
                U_hat[:,i] = np.array([0,0,1,0,0,0])
            elif y[1,i] >= -6*d1 and y[1,i] < -4*d1:  # Punkt 50
                U_hat[:,i] = np.array([0,0,1,0,0,1])
            elif y[1,i] >= -4*d1 and y[1,i] < -2*d1:  # Punkt 36
                U_hat[:,i] = np.array([0,0,1,0,1,1])
            elif y[1,i] >= -2*d1 and y[1,i] < 0:      # Punkt 16
                U_hat[:,i] = np.array([0,0,1,0,1,0])
            elif y[1,i] >= 0 and y[1,i] < 2*d1:       # Punkt 14
                U_hat[:,i] = np.array([0,0,1,1,1,0])
            elif y[1,i] >= 2*d1 and y[1,i] < 4*d1:    # Punkt 34
                U_hat[:,i] = np.array([0,0,1,1,1,1])
            elif y[1,i] >= 4*d1 and y[1,i] < 6*d1:    # Punkt 49
                U_hat[:,i] = np.array([0,0,1,1,0,1])
            else:                                     # Punkt 55
                U_hat[:,i] = np.array([0,0,1,1,0,0])
        elif y[0,i] >= -4*d1 and y[0,i] < -2*d1:        # Firkant 3
            if y[1,i] < -6*d1:                        # Punkt 45
                U_hat[:,i] = np.array([0,1,1,0,0,0])
            elif y[1,i] >= -6*d1 and y[1,i] < -4*d1:  # Punkt 37
                U_hat[:,i] = np.array([0,1,1,0,0,1])
            elif y[1,i] >= -4*d1 and y[1,i] < -2*d1:  # Punkt 30
                U_hat[:,i] = np.array([0,1,1,0,1,1])
            elif y[1,i] >= -2*d1 and y[1,i] < 0:      # Punkt 8
                U_hat[:,i] = np.array([0,1,1,0,1,0])
            elif y[1,i] >= 0 and y[1,i] < 2*d1:       # Punkt 6
                U_hat[:,i] = np.array([0,1,1,1,1,0])
            elif y[1,i] >= 2*d1 and y[1,i] < 4*d1:    # Punkt 29
                U_hat[:,i] = np.array([0,1,1,1,1,1])
            elif y[1,i] >= 4*d1 and y[1,i] < 6*d1:    # Punkt 35
                U_hat[:,i] = np.array([0,1,1,1,0,1])
            else:                                     # Punkt 43
                U_hat[:,i] = np.array([0,1,1,1,0,0])
        else:                                         # Firkant 4 
            if y[1,i] < -6*d1:                        # Punkt 25
                U_hat[:,i] = np.array([0,1,0,0,0,0])
            elif y[1,i] >= -6*d1 and y[1,i] < -4*d1:  # Punkt 17
                U_hat[:,i] = np.array([0,1,0,0,0,1])
            elif y[1,i] >= -4*d1 and y[1,i] < -2*d1:  # Punkt 9
                U_hat[:,i] = np.array([0,1,0,0,1,1])
            elif y[1,i] >= -2*d1 and y[1,i] < 0:      # Punkt 2
                U_hat[:,i] = np.array([0,1,0,0,1,0])
            elif y[1,i] >= 0 and y[1,i] < 2*d1:       # Punkt 1
                U_hat[:,i] = np.array([0,1,0,1,1,0])
            elif y[1,i] >= 2*d1 and y[1,i] < 4*d1:    # Punkt 7
                U_hat[:,i] = np.array([0,1,0,1,1,1])
            elif y[1,i] >= 4*d1 and y[1,i] < 6*d1:    # Punkt 15
                U_hat[:,i] = np.array([0,1,0,1,0,1])
            else:                                     # Punkt 23
                U_hat[:,i] = np.array([0,1,0,1,0,0])    
    else:                                             # Højre halvplan
        if y[0,i] < 2*d1:                             # Firkant 5
            if y[1,i] < -6*d1:                        # Punkt 27
                U_hat[:,i] = np.array([1,1,0,0,0,0])
            elif y[1,i] >= -6*d1 and y[1,i] < -4*d1:  # Punkt 19
                U_hat[:,i] = np.array([1,1,0,0,0,1])
            elif y[1,i] >= -4*d1 and y[1,i] < -2*d1:  # Punkt 11
                U_hat[:,i] = np.array([1,1,0,0,1,1])
            elif y[1,i] >= -2*d1 and y[1,i] < 0:      # Punkt 3
                U_hat[:,i] = np.array([1,1,0,0,1,0])
            elif y[1,i] >= 0 and y[1,i] < 2*d1:       # Punkt 0
                U_hat[:,i] = np.array([1,1,0,1,1,0])
            elif y[1,i] >= 2*d1 and y[1,i] < 4*d1:    # Punkt 5
                U_hat[:,i] = np.array([1,1,0,1,1,1])
            elif y[1,i] >= 4*d1 and y[1,i] < 6*d1:    # Punkt 13
                U_hat[:,i] = np.array([1,1,0,1,0,1])
            else:                                     # Punkt 21
                U_hat[:,i] = np.array([1,1,0,1,0,0])
        elif y[0,i] >= 2*d1 and y[0,i] < 4*d1:          # Firkant 6
            if y[1,i] < -6*d1:                        # Punkt 47
                U_hat[:,i] = np.array([1,1,1,0,0,0])
            elif y[1,i] >= -6*d1 and y[1,i] < -4*d1:  # Punkt 39
                U_hat[:,i] = np.array([1,1,1,0,0,1])
            elif y[1,i] >= -4*d1 and y[1,i] < -2*d1:  # Punkt 31
                U_hat[:,i] = np.array([1,1,1,0,1,1])
            elif y[1,i] >= -2*d1 and y[1,i] < 0:      # Punkt 10
                U_hat[:,i] = np.array([1,1,1,0,1,0])
            elif y[1,i] >= 0 and y[1,i] < 2*d1:       # Punkt 4
                U_hat[:,i] = np.array([1,1,1,1,1,0])
            elif y[1,i] >= 2*d1 and y[1,i] < 4*d1:    # Punkt 28
                U_hat[:,i] = np.array([1,1,1,1,1,1])
            elif y[1,i] >= 4*d1 and y[1,i] < 6*d1:    # Punkt 33
                U_hat[:,i] = np.array([1,1,1,1,0,1])
            else:                                     # Punkt 41
                U_hat[:,i] = np.array([1,1,1,1,0,0])
        elif y[0,i] >= 4*d1 and y[0,i] < 6*d1:          # Firkant 7
            if y[1,i] < -6*d1:                        # Punkt 59
                U_hat[:,i] = np.array([1,0,1,0,0,0])
            elif y[1,i] >= -6*d1 and y[1,i] < -4*d1:  # Punkt 51
                U_hat[:,i] = np.array([1,0,1,0,0,1])
            elif y[1,i] >= -4*d1 and y[1,i] < -2*d1:  # Punkt 38
                U_hat[:,i] = np.array([1,0,1,0,1,1])
            elif y[1,i] >= -2*d1 and y[1,i] < 0:      # Punkt 18
                U_hat[:,i] = np.array([1,0,1,0,1,0])
            elif y[1,i] >= 0 and y[1,i] < 2*d1:       # Punkt 12
                U_hat[:,i] = np.array([1,0,1,1,1,0])
            elif y[1,i] >= 2*d1 and y[1,i] < 4*d1:    # Punkt 32
                U_hat[:,i] = np.array([1,0,1,1,1,1])
            elif y[1,i] >= 4*d1 and y[1,i] < 6*d1:    # Punkt 48
                U_hat[:,i] = np.array([1,0,1,1,0,1])
            else:                                     # Punkt 53
                U_hat[:,i] = np.array([1,0,1,1,0,0])
        else:                                         # Firkant 8 
            if y[1,i] < -6*d1:                        # Punkt 63
                U_hat[:,i] = np.array([1,0,0,0,0,0])
            elif y[1,i] >= -6*d1 and y[1,i] < -4*d1:  # Punkt 58
                U_hat[:,i] = np.array([1,0,0,0,0,1])
            elif y[1,i] >= -4*d1 and y[1,i] < -2*d1:  # Punkt 46
                U_hat[:,i] = np.array([1,0,0,0,1,1])
            elif y[1,i] >= -2*d1 and y[1,i] < 0:      # Punkt 26
                U_hat[:,i] = np.array([1,0,0,0,1,0])
            elif y[1,i] >= 0 and y[1,i] < 2*d1:       # Punkt 20
                U_hat[:,i] = np.array([1,0,0,1,1,0])
            elif y[1,i] >= 2*d1 and y[1,i] < 4*d1:    # Punkt 40
                U_hat[:,i] = np.array([1,0,0,1,1,1])
            elif y[1,i] >= 4*d1 and y[1,i] < 6*d1:    # Punkt 52
                U_hat[:,i] = np.array([1,0,0,1,0,1])
            else:                                     # Punkt 60
                U_hat[:,i] = np.array([1,0,0,1,0,0])
    


# Number of errors
N_error_bit = sum(sum(abs(U_hat-U)))

BEM = abs(U_hat - U)        # Bit error matrix
BCM = np.ones([k,L]) - BEM  # Bit correct matrix
SCM = BCM[0,:]*BCM[1,:]     # Symbol correct matrix

N_error_sym = L - sum(SCM)

pb = N_error_bit / (L*k)
ps = N_error_sym / L

print("BER: ", pb)
print("SER: ", ps)
