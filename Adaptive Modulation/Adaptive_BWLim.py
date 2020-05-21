# -*- coding: utf-8 -*-
"""
This scipt is used to simulate adaptive modulation schemes in base-band via. IQ signals.
It depends on gen_BitMatrix.py, gen_SymCon.py and get_SNRdb.py
"""
__author__ = "EIT6,Group 651"
__credits__ = ["Kasper Bruun Olesen", "Nicolai Almskou Rasmussen", "Victor Mølbach Nissen"]
__email__ = "{kolese17, nara17, vnisse17}@student.aau.dk"


# Packages
import numpy as np # Numerical functions
import matplotlib.pyplot as plt # Plots
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as tck


# Custom packages
import gen_SymCon as SC # Generation of symbol constellations (IQ values)
import gen_BitMatrix as BM # Generation of possible binary valyues
import get_SNRdb as GS # 
import get_Dist as GD

### Parameter
L = int(1e6)           # Number of symbol sent per position (recommend 1e7, but it will take time!)
                       # Bit error rates higher than thre thresholds might arise if < 1e7 symbols are transferred, because the simulations will lack data.
Es = 1                 # Mean symbol energy (keep at 1)
n = 20                # Number of samples in the orbit
P = 1                  # Number of periods (recommended to be kept at 1, or simulation time increases drastically)
BW = 10e03             # Bandwidth

# Parameters for the orbit: [Radius [m], Angle x-axis [rad], Angle z-axis [rad]]
OrbPara = [7571*1000, np.pi/18, np.pi/40]

# Parameters for the sattelites: 
# [Power Trans., Gain Trans., Gain Receiver, Noise Temp., Carrier Freq.]
SatPara = [-3, 7, 7, 21.3, 2269.8e6]

### Variables (be careful if changing these)
T = n*P                # Number of total steps
Rs = BW//2   
# For validation:
deviation_mean_noise_energy = np.zeros(T)
deviation_mean_symbol_energy = np.zeros(T)

### Useful bandwidths to test with:
# BW = 100e03 # BPSK and QPSK
# BW = 50e03 # QPSK and 16 QAM
# BW = 10e03 # 16 QAM and 64 QAM

# Matrix to save values: [SNRdb_bit, pb, ps, troughput, modulation scheme] 
SaveMatrix = np.zeros([5,T])

### Bit rate and SNR values (change according to BER value)
#Thresholds in SNR/sym

# BER = 1e-05
BPSK_thr  = 9.52 + 10*np.log10(1)
QPSK_thr  = 9.58 + 10*np.log10(2)
QAM16_thr = 13.41 + 10*np.log10(4)
QAM64_thr = 17.74 + 10*np.log10(6)

# =============================================================================
# # BER = 1e-04
# BPSK_thr  = 8.35 + 10*np.log10(1)
# QPSK_thr  = 8.36 + 10*np.log10(2)
# QAM16_thr = 12.17 + 10*np.log10(4)
# QAM64_thr = 16.5 + 10*np.log10(6)
# =============================================================================


# Get the SNR values for n equally split values in the orbit
SNRvalues = GS.GetSNRdb_sym(Rs, n, OrbPara, SatPara)

# Create a matrix with the simulation parameters for n simulations
Orbit = np.zeros([3,n]) # Start with 0. If SNR gets under threshold, symbol rate will be 0.
Orbit[2,:] = SNRvalues

# Row 0: Modulation scheme, Row 1: Symbol rate, Row 2: SNR/sym
# BSPK = 0, QPSK = 1, 16-QAM = 2, 64-QAM = 3

# BPSK
Orbit[:-1,SNRvalues > BPSK_thr] = [[0],[Rs]] 

# QPSK
Orbit[:-2,SNRvalues > QPSK_thr] = [1] 

# 16-QAM
Orbit[:-2,SNRvalues > QAM16_thr] = [2] 

# 64-QAM
Orbit[:-2,SNRvalues > QAM64_thr] = [3] 


### Constellations
# Gets the constellations
s_BPSK, s_sort_BPSK = SC.SymbolConstellation('BPSK', Es)
s_QPSK, s_sort_QPSK = SC.SymbolConstellation('QPSK', Es)
s_16QAM, s_sort_16QAM = SC.SymbolConstellation('16QAM', Es)
s_64QAM, s_sort_64QAM = SC.SymbolConstellation('64QAM', Es)
        

# Gets alle possible binary values
b_BPSK = BM.BitMatrix('BPSK')
b_QPSK = BM.BitMatrix('QPSK')
b_16QAM = BM.BitMatrix('16QAM')
b_64QAM = BM.BitMatrix('64QAM')


# Simulate the channel for all SNR values and calculate the BER.
for j in range(T):
    # Prints how far the simulation is
    progress = j/T
    print(f"Progress: {progress*100}%")    
    

### Getting the right Constellation  
    if int(Orbit[0,np.mod(j,n)]) == 0:   # BPSK
        k = 1                            # Number of bits
        s = s_BPSK
        s_sort = s_sort_BPSK
        b = b_BPSK
    elif int(Orbit[0,np.mod(j,n)]) == 1: # QPSK
        k = 2
        s = s_QPSK
        s_sort = s_sort_QPSK
        b = b_QPSK
    elif int(Orbit[0,np.mod(j,n)]) == 2: # 16-QAM
        k = 4
        s = s_16QAM
        s_sort = s_sort_16QAM
        b = b_16QAM
    else:                                # 64 - QAM
        k = 6
        s = s_64QAM
        s_sort = s_sort_64QAM
        b = b_64QAM
        
    # Calculate the number of symbols
    M = 2**k

    # Saves the throughput and which modulation scheme is used
    SaveMatrix[3,j] = k*Orbit[1,np.mod(j,n)]     # Throughput - bit rate    
    SaveMatrix[4,j] = Orbit[0,np.mod(j,n)]       # Modulation scheme      
        
    
    # Variables 
    SNRdb = Orbit[2,np.mod(j,n)] 
    SNR = (10**(SNRdb/10))      # from db to linear
    N0_half = Es/(SNR*2)        # Noise variance (eq 2.38 in report) (N0_half = sigma**2)
    
    # Sample seq. 
    U = np.random.randint(0,2,[L,k]) # Making L*k bits (random bit stream)
    
    # Arrays
    x = np.zeros([L,2])              # The sent signal
    U_hat = np.zeros([k,L])          # The decoded input to receiver
       
      
    # Check mean constellation symbol power:
    total_power = (s[:,:]**2).sum()
    mean_power = total_power/M
   
### Mapping LUT
    # Maps the bit stream to the symbols for transmitting
            
    # For each possible symbol, insert the IQ values in x
    for i in range(M):
        x[np.argwhere(np.all(U == b[i,:], axis=1))] = s_sort[:,i]  
        
    #Calculate mean symbol power (validation)
    total_power_generated = (x[:,:]**2).sum()
    mean_power_generated = total_power_generated/L
    deviation_mean_symbol_energy[j-1] = Es - mean_power_generated
    
### Channel
    
    # Noise
    w = np.random.randn(L,2)*np.sqrt(N0_half)
    
    #Calculate mean noise power (validation)
    total_noise_generated = (w[:,:]**2).sum()
    mean_noise_generated = total_noise_generated/L

    deviation_mean_noise_energy[j-1] = N0_half*2 - mean_noise_generated
          
    # Add noise
    y = x + w
    
### Decode / inv(encode)
    
    # Create a matrix which has the distance from each IQ value set to every symbol
    Dist = np.zeros([L,M])
    for i in range(M):
        Dist[:,i] = (np.sqrt((s_sort[0,i]-y[:,0])**2 + (s_sort[1,i] - y[:,1])**2))
    
    # return the index for the places each min occour in each row and uses them in 'b' (binary value matrix)
    # to put the correct symbol inside U_hat
    U_hat = b[Dist.argmin(axis = 1),:] 
    
    
### ERROR ANALYSIS 
    # Number of errors
    BEM = abs(U_hat - U)        # Bit error matrix
    BCM = np.ones([L,k]) - BEM  # Bit correct matrix
    SCM = np.ones([1,L])        # Symbol correct vector
    for i in range(k):
        SCM = SCM*BCM[:,i]      #If it reaches a 0 down the road, the SCM index becomes a zero
    
    N_error_bit = sum(sum(abs(BEM)))
    
    N_error_sym = L - sum(sum(SCM))
    
    pb = N_error_bit / (L*k)
    ps = N_error_sym / L
    
    # Saves the SNR/sym value, pb and ps
    if pb == 0:
        pb = None #Make sure the 0 wont get plotted
    if ps == 0:
        ps = None
    SaveMatrix[[0,1,2],j] = [SNRdb,pb,ps]

print(f"Progress: 100%")

### Plots
t = np.arange(0, 2 * np.pi * (T/n), 2 * np.pi * (1/n)) #Angle axis for plotting SNR vs angle

# Formatting the plots:
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 12} #Change text size according to screen resolution (14 is best for 4k, otherwise use 10)

plt.rc('font', **font)
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0,0))

# Have increments of pi on the x-axis (ticks must be set in increments of pi/2!):
def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

# Plot the simulated BER through the orbit:
fig1 = plt.figure(figsize=(8,3))
ax = fig1.gca()
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) # Scientific formatting
ax.yaxis.major.formatter._useMathText = True #10^n instead of 1 e n etc
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func)) #Set increments of pi on x-axis
ax.xaxis.set_major_locator(tck.MultipleLocator(base=np.pi/2))
ax.scatter(t[SaveMatrix[4,:]==0],SaveMatrix[1,(SaveMatrix[4,:]==0)],label='BPSK')
ax.scatter(t[SaveMatrix[4,:]==1],SaveMatrix[1,(SaveMatrix[4,:]==1)],label='QPSK')
ax.scatter(t[SaveMatrix[4,:]==2],SaveMatrix[1,(SaveMatrix[4,:]==2)],label='16QAM')
ax.scatter(t[SaveMatrix[4,:]==3],SaveMatrix[1,(SaveMatrix[4,:]==3)],label='64QAM')
ax.set_title(f'Simulated BER Through Orbit, BW = {BW/1000} kHz')
ax.set_yscale('log')
ax.set_ylim(10e-8,10e-5)
ax.legend(loc='upper right')
ax.set_ylabel('Bit Error Rate')
ax.set_xlabel('Satellite Position [rad]')

# Plot the SNR through the orbit:
fig2 = plt.figure(figsize=(8,3))
ax2 = fig2.gca()
ax2.ticklabel_format(axis='y',style='plain',scilimits=(0,0))
ax2.yaxis.major.formatter._useMathText = True
ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax2.xaxis.set_major_locator(tck.MultipleLocator(base=np.pi/2))
ax2.scatter(t[SaveMatrix[4,:]==0],SaveMatrix[0,(SaveMatrix[4,:]==0)],label='BPSK')
ax2.scatter(t[SaveMatrix[4,:]==1],SaveMatrix[0,(SaveMatrix[4,:]==1)],label='QPSK')
ax2.scatter(t[SaveMatrix[4,:]==2],SaveMatrix[0,(SaveMatrix[4,:]==2)],label='16QAM')
ax2.scatter(t[SaveMatrix[4,:]==3],SaveMatrix[0,(SaveMatrix[4,:]==3)],label='64QAM')
ax2.set_title(f'SNR Through Orbit, BW = {BW/1000} kHz')
ax2.legend(loc='upper right')
ax2.set_ylabel('SNR / sym')
ax2.set_xlabel('Satellite Position [rad]')

# Plot the throughput through the orbit:
fig3 = plt.figure(figsize=(8,3))
ax3 = fig3.gca()
ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax3.xaxis.set_major_locator(tck.MultipleLocator(base=np.pi/2))
ax3.scatter(t[SaveMatrix[4,:]==0],SaveMatrix[3,(SaveMatrix[4,:]==0)]//1000,label='BPSK')
ax3.scatter(t[SaveMatrix[4,:]==1],SaveMatrix[3,(SaveMatrix[4,:]==1)]//1000,label='QPSK')
ax3.scatter(t[SaveMatrix[4,:]==2],SaveMatrix[3,(SaveMatrix[4,:]==2)]//1000,label='16QAM')
ax3.scatter(t[SaveMatrix[4,:]==3],SaveMatrix[3,(SaveMatrix[4,:]==3)]//1000,label='64QAM')
ax3.set_title(f'Throughput Through Orbit, BW = {BW/1000} kHz')
ax3.legend(loc='upper right')
ax3.set_ylabel('[kbps]')
ax3.set_xlabel('Satellite Position [rad]')

# Plot the SNR through one orbital period with all modulation scheme thresholds
t_plot = np.arange(0,2*np.pi,2*np.pi*(1/(n)))
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)

ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax4.yaxis.set_major_formatter(formatter)
ax4.locator_params(nbins=10,axis='y')
ax4.xaxis.set_major_locator(tck.MultipleLocator(base=np.pi/2))

ax4.plot(t_plot,SNRvalues,color='black',label='SNR/symbol')
ax4.set_xlabel('Satellite Position [rad]')
ax4.set_ylabel('SNR/symbol [dB]')
ax4.set_title(f'SNR values between satellites, BW = {BW/1000} kHz')

# Fill to show different modulation scheme regions:
ax4.fill_between(t_plot, BPSK_thr, SNRvalues,where=SNRvalues >= BPSK_thr,color='#e5e0e4', label='BPSK')
ax4.fill_between(t_plot, QPSK_thr, SNRvalues,where=SNRvalues >= QPSK_thr,color='#c3c8cc', label='QPSK')
ax4.fill_between(t_plot, QAM16_thr, SNRvalues,where=SNRvalues >= QAM16_thr,color='#9db1ba', label='16-QAM')
ax4.fill_between(t_plot, QAM64_thr, SNRvalues,where=SNRvalues >= QAM64_thr,color='#839eaa', label='64-QAM')
ax4.legend(loc = 'lower left')

# Plot horizontal lines to make the border between regions more clear:
QAM_64_hori_plot = np.full(n,QAM64_thr)
QAM_64_hori_plot[SNRvalues<=QAM64_thr] = None
QAM_16_hori_plot = np.full(n,QAM16_thr)
QAM_16_hori_plot[SNRvalues<=QAM16_thr] = None
QPSK_hori_plot = np.full(n,QPSK_thr)
QPSK_hori_plot[SNRvalues<=QPSK_thr] = None
BPSK_hori_plot = np.full(n,BPSK_thr)
BPSK_hori_plot[SNRvalues<=BPSK_thr] = None

ax4.plot(t_plot,QAM_64_hori_plot,linewidth=1,color='#000000')
ax4.plot(t_plot,QAM_16_hori_plot,linewidth=1,color='#000000')
ax4.plot(t_plot,QPSK_hori_plot,linewidth=1,color='#000000')
ax4.plot(t_plot,BPSK_hori_plot,linewidth=1,color='#000000')



full_distance_array = GD.GetDistance(n)
divider = 4 #Leave this at 4, or the script implodes
one_half_cycle = full_distance_array[1:(n//divider)+1:1] #Holds the minimum and maximum distance

SNR = GS.GetSNRdb_sym(Rs, n) # Gets the SNR around the whole orbit
SNR_shorted = SNR[1:(n//divider)+1:1] # We are only interested in the first quarter of the values

#Arrays for plotting horisontal lines:
QAM_64_hori_plot = np.full(n//divider,QAM64_thr)
QAM_64_hori_plot[SNR_shorted<QAM64_thr] = None
QAM_16_hori_plot = np.full(n//divider,QAM16_thr)
QAM_16_hori_plot[SNR_shorted<QAM16_thr] = None
QPSK_hori_plot = np.full(n//divider,QPSK_thr)
QPSK_hori_plot[SNR_shorted<QPSK_thr] = None
BPSK_hori_plot = np.full(n//divider,BPSK_thr)
BPSK_hori_plot[SNR_shorted<BPSK_thr] = None

# Plot the same plot as before, but with the distance on the x-axis instead of angle:
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.plot(one_half_cycle,SNR[1:(n//divider)+1:1],color='#000000',label='SNR/symbol')

ax5.plot(one_half_cycle,QAM_64_hori_plot,linewidth=1,color='#000000')
ax5.plot(one_half_cycle,QAM_16_hori_plot,linewidth=1,color='#000000')
ax5.plot(one_half_cycle,QPSK_hori_plot,linewidth=1,color='#000000')
ax5.plot(one_half_cycle,BPSK_hori_plot,linewidth=1,color='#000000')

ax5.fill_between(one_half_cycle, BPSK_thr, SNR_shorted,where=SNR_shorted >= BPSK_thr,color='#e5e0e4', label='BPSK')
ax5.fill_between(one_half_cycle, QPSK_thr, SNR_shorted,where=SNR_shorted >= QPSK_thr,color='#c3c8cc', label='QPSK')
ax5.fill_between(one_half_cycle, QAM16_thr, SNR_shorted,where=SNR_shorted >= QAM16_thr,color='#9db1ba', label='16-QAM')
ax5.fill_between(one_half_cycle, QAM64_thr, SNR_shorted,where=SNR_shorted >= QAM64_thr,color='#839eaa', label='64-QAM')

ax5.legend(loc = 'lower left')


ax5.yaxis.set_major_formatter(formatter)
ax5.xaxis.set_major_formatter(formatter)
ax5.locator_params(nbins=9,axis='y')

ax5.set_ylabel('SNR/symbol [dB]')
ax5.set_xlabel('Distance between satellites [m]')
ax5.set_title(f'Modulation scheme regions for BW = {BW/1000} kHz')

### Validation of simulations:
plt.tight_layout()
step_axis = np.arange(1,T+1,1)
fig6 = plt.figure(figsize=(8,3))
ax6 = fig6.gca()
ax6.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) # Scientific formatting
ax6.yaxis.major.formatter._useMathText = True #10^n instead of 1 e n etc
ax6.plot(step_axis, deviation_mean_noise_energy,label="Deviation from expected noise energy")
ax6.legend(loc = 'lower left')
ax6.set_xlabel('Simulation run')
ax6.set_ylabel('Noise deviation')

fig7 = plt.figure(figsize=(8,3))
ax7 = fig7.gca()
ax7.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) # Scientific formatting
ax7.yaxis.major.formatter._useMathText = True #10^n instead of 1 e n etc
ax7.plot(step_axis, deviation_mean_symbol_energy,label="Deviation from expected symbol energy")
ax7.legend(loc = 'lower left')
ax7.set_xlabel('Simulation run')
ax7.set_ylabel('Symbol deviation')
