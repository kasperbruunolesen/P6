# -*- coding: utf-8 -*-
"""
Different function which handles the calculations with SNR and symbol/bit rate.

GetDistance(n, p)
    n: Number of point. int
    p: Parameters for the orbit: 
        [Radius [m], Angle x-axis [rad], Angle z-axis [rad]]
        values for the oneweb constellation:
            [7571*1000, 2 * 3.14 * (10/360), 2 * 3.14 * (4.5/360)]
    Return a array with the distances in [m] for one period of data
"""

__author__ = "EIT6,Group 651"
__credits__ = ["Kasper Bruun Olesen", "Nicolai Almskou Rasmussen", "Victor Mølbach Nissen"]
__email__ = "{kolese17, nara17, vnisse17}@student.aau.dk"

def GetDistance(n, p = [7571*1000, 2 * 3.14 * (10/360), 2 * 3.14 * (4.5/360)]):
    import numpy as np

    # radius [m]
    r = p[0]
    
    # Angles [rad]:
    a_x = p[1]
    a_z = p[2]
    
    # Rotation Matrices
    Rot_x = lambda a: [[1,0,0],
                       [0,np.cos(a),-np.sin(a)],
                       [0,np.sin(a),np.cos(a)]]
    
    Rot_z = lambda a: [[np.cos(a),-np.sin(a),0],
                       [np.sin(a),np.cos(a),0],
                       [0,0,1]]
        
    # Creating the orbits
    t = np.arange(n)[np.newaxis].reshape(-1)
    C1 = np.vstack([r*np.cos(2 * np.pi * (t/n)),r*np.sin(2 * np.pi * (t/n)),np.zeros(n)])
    C2 = np.dot(Rot_x(a_x),np.dot(Rot_z(a_z),C1)) #Rot_x(a_x)*(Rot_z(a_z)*C1)
    
    # Calculating the distance with norm 2
    Dist = np.linalg.norm(C1-C2,2,axis = 0)
    
    return Dist 
