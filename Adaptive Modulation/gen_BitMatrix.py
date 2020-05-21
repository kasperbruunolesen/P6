# -*- coding: utf-8 -*-
"""
This is a module for generating the a bit matrix for different modulations schemes
BitMatrix(modulation):
    modulation: name of the modulation scheme, 
        string: 'BPSK', 'QPSK', '16QAM' and '64QAM'
    Return the bit matrix .
"""

__author__ = "EIT6,Group 651"
__credits__ = ["Kasper Bruun Olesen", "Nicolai Almskou Rasmussen", "Victor Mølbach Nissen"]
__email__ = "{kolese17, nara17, vnisse17}@student.aau.dk"


def BitMatrix(modulation):
    import numpy as np
    
    if modulation == 'BPSK':
        k = 1
        M = 2**k
        b = np.zeros([M,k])
        for i0 in range(2):
            b[i0,:] = [i0]
        
    elif modulation == 'QPSK':
        k = 2
        M = 2**k
        b = np.zeros([M,k])
        for i1 in range(2):
            for i0 in range(2):
                b[2*i1 + i0,:] = [i1,i0]        
    
    elif modulation == '16QAM':
        k = 4
        M = 2**k
        b = np.zeros([M,k])
        for i3 in range(2):
            for i2 in range(2):
                for i1 in range(2):
                    for i0 in range(2):
                        b[8*i3 + 4*i2 + 2*i1 + i0,:] = [i3,i2,i1,i0]          
               
    else:
        k = 6
        M = 2**k
        b = np.zeros([M,k])
        for i5 in range(2):
            for i4 in range(2):
                for i3 in range(2):
                    for i2 in range(2):
                        for i1 in range(2):
                            for i0 in range(2):
                                b[32*i5 + 16*i4 + 8*i3 + 4*i2 + 2*i1 + i0,:] = [i5,i4,i3,i2,i1,i0]        

    return b    

