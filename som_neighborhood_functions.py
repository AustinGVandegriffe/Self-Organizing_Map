"""
/// @file som_neighborhood_functions.py
/// @author Austin Vandegriffe
/// @date 2020-02-23
/// @brief Metric class for the self organizing map.
/// @style K&R, and "one true brace style" (OTBS), and '_' variable naming
/////////////////////////////////////////////////////////////////////
/// @references
/// ## [1] http://www.cis.hut.fi/projects/somtoolbox/package/papers/techrep.pdf
/// ## [2] https://community.jmp.com/t5/JMP-Blog/Self-organizing-maps/ba-p/224917
/// @notes
/// ## 1. SOM.py depends on this file.
"""

import numpy as np
from types import FunctionType as FunctionPtr

class Neighborhood_Functions(object):
    """
    [1] top of page 10
    """
    def __init__(self, 
                    sigma_t :float = 3, 
                    T       :int   =250,
                    end     :int   =float("inf")
    ) -> None:
        """
            sigma_t := neighborhood radius at time t (i.e. function pointer)
        """
        self.T = T
        if type(sigma_t) in [int, float]:
            self.sigma_t = lambda t: sigma_t*np.exp(-t/T)
        elif type(sigma_t) == FunctionPtr:
            self.sigma_t = sigma_t
        else:
            raise TypeError("Invalid sigma_t ...")
        self.end = end
    
    def bubble(self, 
                bmu :np.array, 
                i   :np.array, 
                t   :int
    ) -> float:
        if t < self.end:
            g_dist = np.linalg.norm(bmu-i)
            return int( (sigma_t(t) - g_dist) < 0)
        else:
            return int(np.prod(bmu == i))

    def gaussian(self, 
                    bmu :np.array, 
                    i   :np.array, 
                    t   :int
    ) -> float:
        if t < self.end:
            g_dist = np.linalg.norm(bmu-i)
            return np.exp( -( g_dist / self.sigma_t(t) ) )
        else:
            return int(np.prod(bmu == i))

    def cutgauss(self, 
                    bmu :np.array, 
                    i   :np.array, 
                    t   :int
    ) -> float:
        """
            Just self.gaussian * self.bubble
        """
        if t < self.end:
            g_dist = np.linalg.norm(bmu-i)
            return np.exp( -( g_dist / self.sigma_t(t) ) ) * int( (self.sigma_t(t) - g_dist) < 0)
        else:
            return int(np.prod(bmu == i))
    
    def ep(self, 
                bmu :np.array, 
                i   :np.array, 
                t   :int
    ) -> float:
        if t < self.end:
            g_dist = np.linalg.norm(bmu-i)
            return np.max([0, 1 - (self.sigma_t(t) - g_dist)**2 ])
        else:
            return int(np.prod(bmu == i))
