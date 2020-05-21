# -*- coding: utf-8 -*-
"""
Different function which handles the calculations with SNR and symbol/bit rate.

GetSNRdb_sym(Rs, n, p, s)
    Rs: Baud rate, float / float array.
    n: Number of parts in the orbit, int.
    p: Parameters for the orbit: 
        [Radius [m], Angle x-axis [rad], Angle z-axis [rad]]
        values for the oneweb constellation:
            [7571*1000, 2 * 3.14 * (10/360), 2 * 3.14 * (4.5/360)]
    s: Parameters for the satellite: 
        [Power Trans., Gain Trans., Gain Receiver, Noise Temp., Carrier Freq.]
        values for the S-Net satellites:
            [-3, 7, 7, 21.3, 2269.8e6]
    Return the symbol SNR based on the given symbol rate and the position (returns budget for modulation). 

GetSNRdb_no_sym_rate(n,p,s)
    n: Number of parts in the orbit, int.
    p: Parameters for the orbit: 
        [Radius [m], Angle x-axis [rad], Angle z-axis [rad]]
        values for the oneweb constellation:
            [7571*1000, 2 * 3.14 * (10/360), 2 * 3.14 * (4.5/360)]
    s: Parameters for the satellite: 
        [Power Trans., Gain Trans., Gain Receiver, Noise Temp., Carrier Freq.]
        values for the S-Net satellites:
            [-3, 7, 7, 21.3, 2269.8e6]
    Return the SNR without the loss from the baud rate bases on the position (returns budget for modulation and sym rate).

GetBitRate(SNRdB_bit,Threshold)
    SNRdb_bit: the SNR in decibel per bit, float / float  array.
    Threshold: The SNR (per bit!) threshold before the BER gets higher than wanted.
    Return the bit rate for the given SNR values bases on the given threshold.

GetSymRate(SNRdB,k,Threshold)
    SNRdb: the SNR (per symbol), float / float  array.
    k: number of bits per symbol. 
    Threshold: The SNR in decibel per bit threshold before the BER gets higher than wanted.
    Return the baudrate for the given SNR values bases on the given threshold   
"""

__author__ = "EIT6,Group 651"
__credits__ = ["Kasper Bruun Olesen", "Nicolai Almskou Rasmussen", "Victor Mølbach Nissen"]
__email__ = "{kolese17, nara17, vnisse17}@student.aau.dk"



def GetSNRdb_sym(Rs, n, 
                 p = [7571*1000, 2 * 3.14 * (10/360), 2 * 3.14 * (4.5/360)],
                 s = [-3, 7, 7, 21.3, 2269.8e6]):
    import numpy as np
    import get_Dist as GD

    # Get Distance in [m]
    Dist = GD.GetDistance(n, p)

    # Constants [dB]
    Pt = s[0]
    Gt = s[1]
    Gr = s[2]
    T  = s[3] 
    fc = s[4]
    kb = -228.6
    c = 3e8
      
    
    # Pathloss
    L = 10*np.log10((4 * np.pi * Dist * fc / c)**2)
    
    
    SNRdB_sym = np.zeros(n)
    
    if type(Rs) is int or type(Rs) is float:
        SNRdB_sym = Pt+Gt+Gr-L-kb-T-10*np.log10(Rs)
    else:
        SNRdB_sym[Rs > 0] = Pt+Gt+Gr-L[Rs > 0]-kb-T-10*np.log10(Rs[Rs > 0])
        SNRdB_sym[Rs == 0] = Pt+Gt+Gr-L[Rs == 0]-kb-T # log(0) is ill-defined, will give infinite SNR

    return SNRdB_sym

def GetSNRdb_no_sym_rate(n, 
                         p = [7571*1000, 2 * 3.14 * (10/360), 2 * 3.14 * (4.5/360)],
                         s = [-3, 7, 7, 21.3, 2269.8e6]):
    import numpy as np
    import get_Dist as GD

    # Get Distance in [k]
    Dist = GD.GetDistance(n , p)
    

    # Constants [dB]
    Pt = s[0]
    Gt = s[1]
    Gr = s[2]
    T  = s[3] 
    fc = s[4]
    kb = -228.6
    c = 3e8
    
    
    # Pathloss
    L = 10*np.log10((4 * np.pi * Dist * fc / c)**2)
    
    
    SNRdB = np.zeros(n)
    SNRdB = Pt+Gt+Gr-L-kb-T

    return SNRdB


def GetBitRate(SNRdB_bit,Threshold):

    Rb = (10**((SNRdB_bit-Threshold)/(10)))
    
    return Rb

    
def GetSymRate(SNRdB_sym,k,Threshold):

    Rs = (10**((SNRdB_sym-Threshold)/(10)))/k
    
    return Rs
    