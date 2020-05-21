# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:38:53 2020

@author: EIT6, Group 651
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt

### Parameters
Fs = 24000        # Sample Frequency [Hz]
fc = 2400        # Carrier Frequency  [Hz]
Fbit = 1200      # Bitrate [Hz]
Cutoff = 3000     # Cutoff Frequency
fd = 0 #0.1          # Dobbler frequency [Hz]
L = 10          # No. of symbols
k = 1           # No. of bits per symbol
M = 2**k        # No. of different symbols

# Choose the wanted plot (unfortunately only 1 at the time):
plot = True  # Sine wave
plot2 = False # IQ constellations
plot3 = False # I_hat and Q_hat
plot4 = False  #IQ sampled 


# Noise
Es = 1
SNRdb_bit = 100  # Signal-noise ratio db
SNR = (10**(SNRdb_bit/10))*k  # from db to linear
N0_half = Es/(SNR * 2)  # Noise variance (eq 2.38 in report) (N0_half = sigma**2)

#Plot stuff:
#plt.clf() #Clear the existing figure if open
font = {'family' : 'normal',
        'size'   : 12}

plt.rc('font', **font) #Set font size for legends etc

# Random bit sequence
U = np.random.randint(0,2,[k,L])
# Non random bit sequence:
#U = np.array([[1,0,1,0,1,0,1,0,1,0]])

### Constellation
s = np.array([1 + 0j, -1 + 0j])


# Expand the constellation 
U_exp = U.flatten() #Make it one dimension smaller
U_exp[U_exp == 0] = -1 #-1 so that U_exp can be used for plotting
U_exp = U_exp.repeat(Fs // Fbit)  # Repeat so that there are Fs/Fbit copies of every symbol. // = return largest possible integer

s_exp = np.zeros_like(U_exp,dtype=complex) #Make a matrix the same size of U_exp with complex numbers in it
s_exp[U_exp == 1] = s[0] #For every one, input 1+0j
s_exp[U_exp == -1] = s[1] #For every -1, input -1+0j

# Plot2 = IQ constellations
if plot2:
    plt.subplot(4,1,1)
    plt.scatter(s_exp.real, s_exp.imag, label = 's_exp')
    plt.legend(loc='upper right')
    plt.title("Constellation")


### Channel
t = np.arange(1/Fs*0.25,L/Fbit,1/Fs) # 1/Fs = halv omgang - ganges med 0.25 for at flytte det væk fra "0".

# Create the signal from the transmitter
x = 1*s_exp.real*np.cos(2*np.pi*fc*t) + 1*s_exp.imag*np.sin(2*np.pi*fc*t)

# Noise
w = np.random.randn(int(L*Fs/Fbit))*np.sqrt(N0_half)

# Recieved signal:
y = x + w

# Plot
if plot:
    plt.subplot(3,1,1)
    #plt.plot(t,x.real, label = 'x.real')
    # plt.plot(t,w, label = 'w')
    plt.plot(t,y, label = 'y')
    plt.legend(loc='upper right')
    plt.title("Signal channel")


### Receiver 

I_hat = 2*(np.cos(2*np.pi*(fc+fd)*t)*y) # Multiply with local frequency, gets the I value 
Q_hat = 2*(np.sin(2*np.pi*(fc+fd)*t)*y) # Multiply with local frequency shifted 90degree, gets the Q value

#I_hat = np.cos(2*np.pi*(fc+fd)*t)*abs(y) # Multiply with local frequency, gets the I value 
#Q_hat = np.sin(2*np.pi*(fc+fd)*t)*abs(y) # Multiply with local frequency shifted 90degree, gets the Q value



# Skal skaleres op - Normalisering. Der forsvinder noget amplitude når der ganges sin og cos på.

# Plot
if plot:
    plt.subplot(3,1,2)
    plt.plot(t,I_hat, label = 'I_hat')
    plt.plot(t,Q_hat, label = 'Q_hat')
    plt.legend(loc='upper right')
    plt.title("Signal received")

if plot2:
    plt.subplot(4,1,2)
    plt.scatter(I_hat,Q_hat, label = 'hat')
    plt.legend(loc='upper right')
    plt.title("IQ receiver")

# Lowpass filter
normal_cutoff = Cutoff/(Fs * 0.5)
b, a = butter(4, normal_cutoff, btype='low', analog=False)
I_lp = filtfilt(b, a, I_hat)
Q_lp = filtfilt(b, a, Q_hat)
# raised cosine filter skal bruges

if plot3:
    plt.scatter(I_lp,Q_lp, label = 'Low pass filtered')
    plt.legend(loc='upper right')
    plt.scatter(I_hat,Q_hat, label = 'Unfiltered signal')
    plt.legend(loc='upper right')
    plt.title("IQ receiver after filter")     

# Plot
if plot:
    plt.subplot(3,1,3)
    plt.plot(t,I_lp, label = 'I_lp')
    plt.plot(t,Q_lp, label = 'Q_lp')
    plt.plot(t,U_exp, label = 'U')
    plt.legend(loc='upper right')
    plt.title("Filtered signal")
if plot2:
    plt.subplot(4,1,3)
    plt.scatter(I_lp,Q_lp, label = 'lp')
    plt.legend(loc='upper right')  
    plt.title("IQ lp")

### Estimated bit sequence
Q_hat_lp = np.zeros(L)
I_hat_lp = np.zeros(L)
U_hat = np.zeros(L)
Q_Temp = np.zeros(L) # Used only in plot
I_Temp = np.zeros(L) # Used only in plot

# Demodulation
sample = np.arange(int(Fs/Fbit/2),(L)*int(Fs/Fbit)+int(Fs/Fbit/2),int(Fs/Fbit))
Q_Temp = Q_lp[sample]
I_Temp = I_lp[sample]
I_hat_lp = np.where(I_lp[sample] >= 0, 1, -1)
Q_hat_lp = np.where(Q_lp[sample] >= 0, 0, 0)
U_hat[I_hat_lp == 1] = 1

# =============================================================================
# for i in range(L):
#     #Q_Temp[i] = Q_lp[i*int(Fs/Fbit)+int(Fs/Fbit/2)] # Used only in plot
#     #I_Temp[i] = I_lp[i*int(Fs/Fbit)+int(Fs/Fbit/2)] # Used only in plot
#     Q_hat_lp[i] = 0
#     if I_lp[i*int(Fs/Fbit)+int(Fs/Fbit/2)] >= 0:
#         I_hat_lp[i] = 1
#     else:
#         I_hat_lp[i] = -1
#     if Q_hat_lp[i] == 0 and I_hat_lp[i] == 1:
#         U_hat[i] = 1
#     else:
#         U_hat[i] = 0
# =============================================================================

# Plot
if plot2:
    plt.subplot(4,1,4)
    plt.scatter(I_Temp,Q_Temp, label = 'Temp')
    plt.legend(loc='upper right')
    plt.title("IQ sample")
    
if plot4:
    plt.scatter(I_Temp,Q_Temp, label = f'fd = {fd}')
    plt.legend(loc='upper right')
    plt.title(f"BPSK, SNR: ∞, No. of symbols: {L}, fc: {fc}")
    plt.ylim(-1.2,1.2)
    plt.xlim(-1.2,1.2)
    plt.xlabel('I')
    plt.ylabel('Q')
        
### Number of errors
BEM = abs(U_hat - U)        # Bit error matrix
N_error_bit = sum(sum(BEM))
BCM = np.ones([2,L]) - BEM  # Bit correct matrix
SCM = BCM[0,:]*BCM[1,:]     # Symbol correct matrix

N_error_sym = L - sum(SCM)

pb = N_error_bit / (L*k)
ps = N_error_sym / L

print("Number of symbols", L)
print("Number of bits", L*k)
print("Number of symbol errors", N_error_sym)
print("Number of bit errors", N_error_bit)
print("BER: ", pb)
print("SER: ", ps)

plt.tight_layout()


# Ting at plotte:
# I_hat sammen med I_lp
# Scatterplot af de steder vi måler ind i t
    # idx = np.arange(L) * int(Fs/Fbit)+int(Fs/Fbit/2)
    # t[idx]
    # Q_lp[idx]