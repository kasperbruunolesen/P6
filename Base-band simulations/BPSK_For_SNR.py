# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:33:12 2020

@author: EIT6, Group 651
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt

### Parameter
L = 100000       # Length (Monte Carlo runs)
Es = 1          # Symbol energy (technically not needed, as long as working with SNR)
k = 1           # Number of bits
M = 2**k        # Number of symbols

SNRdb_max = 20      # Signal-noise radio db
SNR_res = 1         # 1/res per step
SNR_start = 1      # Default 1.  Can be a number between 1 and SNRdb_max * SNR_res. Integer
# Eks. SNR_res = 2, SNRdb_max = 10, Want to start at SNRdb  = 7.5 -> SNR_start = SNR_res*7.5 = 15

# Filename = input("Write Filename: \n")
Filename = "BPSK.txt"
OverWrite = True

# Export data
SaveMatrix = np.zeros([3,(SNRdb_max*SNR_res-SNR_start +1)]) # Matrice which exportes
if OverWrite == True:
    file = open(Filename,"w") 
else:
    file = open(Filename,"a") 
file.write(f"SNR,pb,ps\n")
file.close()

### Constellation
s = np.zeros([2,M]) # For the basis functions
s[:,0]= np.array([1,0])
s[:,1]= np.array([-1,0])

# Constellation Sorted
# Not really need for BPSK but remain so each script is consistence 
s_sort = np.zeros_like(s)
s_sort[:,0] = s[:,0]  # '0' -> s1
s_sort[:,1] = s[:,1]  # '1' -> s1 
    
# Check mean symbol power:
total_power = (s[:,:]**2).sum()
mean_power = total_power/M
print("The mean symbol power is", mean_power)       

### Base symbols
b = np.zeros([M,k])
for i0 in range(2):
    b[i0,:] = [i0]



for SNRdb_bit in range(SNR_start,SNRdb_max*SNR_res+1):
    # Variables 
    SNR = (10**((SNRdb_bit/SNR_res)/10))*k     # from db to linear
    N0_half = Es/(SNR*2)          # Noise variance (eq 2.38 in report) (N0_half = sigma**2)
    N_error_bit = 0               # Number of error bits declaration
    N_error_sym = 0               # Number of error symbols declaration
    
    #Validation variables
    total_noise_generated = 0; # For validation purposes
    total_power_generated = 0; # -||-    
    
    
    # Sample seq. 
    U = np.random.randint(0,2,[L,k]) # Making L*k bits (random bit stream)
    
    # Arrays
    x = np.zeros([L,2])              # The sent signal
    U_hat = np.zeros([k,L])          # The decoded input to receiver
    
        
### Mapping LUT
    # Maps the bit stream to the symbols for transmitting
    # Checks if bit combination if it is in U and return a array with the places index number. 
    # The index number is used to replace the given places in x with the correlated symbol
    for i in range(M):
        x[np.argwhere(np.all(U == b[i,:], axis=1))] = s_sort[:,i]  
    
    
    #Calculate mean symbol power (validation)
    total_power_generated = (x[:,:]**2).sum()
    mean_power_generated = total_power_generated/L
    #print("The mean generated symbol power is", mean_power_generated) 

### Channel
    # Noise
    w = np.random.randn(L,2)*np.sqrt(N0_half)
    
    #Calculate mean noise power (validation)
    total_noise_generated = (w[:,:]**2).sum()
    mean_noise_generated = total_noise_generated/L
    #print("The mean generated noise power is", mean_noise_generated)
    #print("The expected noise power is", N0_half*2) 
          
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
    
    print("")
    print("SNR: ", SNRdb_bit/SNR_res, " of ", SNRdb_max)
    print("Number of symbols", L)
    print("Number of symbol errors", N_error_sym)
    print("Number of bits", L*k)
    print("Number of bit errors", N_error_bit)
    
    print("Validation:")
    print("The mean generated symbol power is", mean_power_generated)
    print("The expected symbol power is", Es)
    print("The mean generated noise power is", mean_noise_generated)
    print("The expected noise power is", 2*N0_half)    
    
    print("BER: ", pb)
    print("SER: ", ps)   
    SaveMatrix[:,SNRdb_bit-SNR_start] = np.array([SNRdb_bit/SNR_res, pb, ps])
    file = open(Filename,"a") 
    file.write(f"{SaveMatrix[0,SNRdb_bit-SNR_start]},{SaveMatrix[1,SNRdb_bit-SNR_start]},{SaveMatrix[2,SNRdb_bit-SNR_start]}\n")
    file.close()

plt.scatter(SaveMatrix[0,:],SaveMatrix[1,:],label = 'pb')
plt.scatter(SaveMatrix[0,:],SaveMatrix[2,:],label = 'ps')
plt.title("BPSK")
plt.xlabel('SNR', color='#1C2833')
plt.ylabel('Error precentage', color='#1C2833')
plt.legend(loc='upper right')
fig = plt.gcf()
# fig.savefig("BPSK.svg")
plt.show()   
    
