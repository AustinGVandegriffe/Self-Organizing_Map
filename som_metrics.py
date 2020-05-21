"""
/// @file som_metrics.py
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

import scipy.spatial.distance as py_distances

class Metrics(object):
    Minkowski = lambda x,y,p : py_distances.minkowski(x, y, p=p)
    Euclidean = lambda x,y   : Metrics.Minkowski(x, y, 2)