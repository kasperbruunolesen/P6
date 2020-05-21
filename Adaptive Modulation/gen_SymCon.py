# -*- coding: utf-8 -*-
"""
This is a module for generating the symbol constellation for different modulation schemes
SymbolConstellation(modulation,Es):
    modulation: name of the modulation scheme, 
        string: 'BPSK', 'QPSK', '16QAM' and '64QAM'
    Es: Symbol energy, float
    Return the symbol constellation and a sorted one. 

"""
__author__ = "EIT6,Group 651"
__credits__ = ["Kasper Bruun Olesen", "Nicolai Almskou Rasmussen", "Victor Mølbach Nissen"]
__email__ = "{kolese17, nara17, vnisse17}@student.aau.dk"


def SymbolConstellation(modulation = 'BPSK',Es = 1):
    import numpy as np
    
    if modulation == 'BPSK':
        k = 1
        M = 2**k
        s = np.zeros([2,M]) # For the basis functions
        s[:,0]= np.array([np.sqrt(Es),0])
        s[:,1]= np.array([-np.sqrt(Es),0])
        
        # Constellation Sorted
        # Not really need for BPSK but remain so each script is consistence 
        s_sort = np.zeros_like(s)
        s_sort[:,0] = s[:,0]  # '0' -> s1
        s_sort[:,1] = s[:,1]  # '1' -> s1
        
    elif modulation == 'QPSK':
        k = 2
        M = 2**k
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
    
    elif modulation == '16QAM':
        k = 4
        M = 2**k
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
        
        
    else:
        k = 6
        M = 2**k
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

    return s,s_sort