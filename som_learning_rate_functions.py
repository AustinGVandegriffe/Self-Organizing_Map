"""
/// @file som_learning_rate_functions.py
/// @author Austin Vandegriffe
/// @date 2020-02-23
/// @brief Learning rate class for the self organizing map.
/// @style K&R, and "one true brace style" (OTBS), and '_' variable naming
/////////////////////////////////////////////////////////////////////
/// @references
/// ## [1] http://www.cis.hut.fi/projects/somtoolbox/package/papers/techrep.pdf
/// ## [2] https://community.jmp.com/t5/JMP-Blog/Self-organizing-maps/ba-p/224917
/// @notes
/// ## 1. SOM.py depends on this file.
"""

class Learning_Rate_Functions(object):
    """
    [1] bottom of page 10.
    """
    def __init__(self, 
                    alpha0:float = 1, 
                    T:int = 1000
    ) -> None:
        """
            T := training length
            alpha0 := initial learning rate
        """
        self.alpha0 = alpha0
        self.T = T

    def linear(self, 
                t:int
    ) -> float:
        if t < self.T:
            return self.alpha0 * ( 1- t/self.T )
        else:
            return 0.01

    def power(self, 
                t:int
    ) -> float:
        if t < self.T:
            return self.alpha0 * ( 0.005/self.alpha0 )**( t/self.T )
        else:
            return 0.01
        
    def inv(self,
                t:int
    ) -> float:
        if t < self.T:
            return self.alpha0 / ( 1 + 100*(t/self.T) )
        else:
            return 0.01