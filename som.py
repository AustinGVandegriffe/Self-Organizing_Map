"""
/// @file som.py
/// @author Austin Vandegriffe
/// @date 2020-02-23
/// @brief Python implementation of the standard self organizing maps,
/// ## for my Spring 2020 presentation for clustering algorithms.
/// @style K&R, and "one true brace style" (OTBS), and '_' variable naming
/////////////////////////////////////////////////////////////////////
/// @references
/// ## [1] http://www.cis.hut.fi/projects/somtoolbox/package/papers/techrep.pdf
/// ## [2] https://community.jmp.com/t5/JMP-Blog/Self-organizing-maps/ba-p/224917
"""

from types import FunctionType as FunctionPtr
import numpy as np
# import joblib # parallel processing to come.
import seaborn as sns

import time
import copy
import random

from som_metrics import *
from som_neighborhood_functions import *
from som_learning_rate_functions import *

def unit(x:np.array) -> np.array:
    return x / np.linalg.norm(x)

class Planer_SOM(object):
    #->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def __init__(self,
                    t_n_rows                :int, 
                    t_n_cols                :int, 
                    dim                     :int,
                    neighborhood_function   :FunctionPtr,
                    lr_function             :FunctionPtr
    ) -> None:
        self.n_rows         : int       = t_n_rows # Number or grid rows
        self.n_cols         : int       = t_n_cols # Number of grid columns
        self.dim            : int       = dim      # Dimension of grid weights
        # Initializing a 2-D plane in \R^{dim}
        ## but beware of topological errors...
        self.grid           : np.array  = np.random.random((t_n_rows, t_n_cols, dim))
        self.differences    : np.array  = np.zeros(self.grid.shape)

        self.data   : np.array  = None # Assigned in SOM.fit

        self.h      :FunctionPtr = neighborhood_function
        self.alpha  :FunctionPtr = lr_function

        self.history = [] # tracks the training process for visualization

        return
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<-
    #->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def sequential_fit(self,
                        data    :np.array, 
                        epochs  :int  = 500000, 
    ) -> None:
        """
            Sequentially fits the SOM to the data.

            Data should have observations vertically and variables horizontally
                i.e. data = [o1, o2, o3, ...].T
        """
        self.data = data
        for t in range(epochs):
            d = data[np.random.choice(self.data.shape[0], size=1)]
            differences = self.grid - d
            winner = np.linalg.norm(differences, axis=2).argmin()
            winner_index = (winner // self.n_rows, winner % self.n_cols)
            delta_grid : np.array = np.zeros(self.grid.shape)

            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    delta_grid[i,j] += ( self.h(np.array(winner_index),np.array((i,j)),t) * differences[i,j] )
            self.grid -= self.alpha(t)*delta_grid

            self.history.append(copy.deepcopy(self.grid))

        return
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<-
    #->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def batch_fit(self,
                        data        :np.array, 
                        epochs      :int = 500000,
                        batch_size : int = 50
    ) -> None:
        '''
            Fit the SOM to the data in bactch mode.

            NOT YET IMPLEMENTED.
        '''
        raise NotImplementedError("Batch mode not yet implemented for the SOM class.")
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<-
    #->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def umatrix(self, 
                    grid:np.array   =None, 
                    plot:bool       =True
    ) -> np.array:
        """
            Plots the UMatrix of SOM.Grid
        """
        if not grid:
            grid = self.grid
        
        matrix = -np.ones((grid.shape[0]*2+1, grid.shape[1]*2+1))
        for r in range(1,matrix.shape[0]-3,2):
            for c in range(1,matrix.shape[1]-3,2):
                t = grid[r//2:r//2+2, c//2:c//2+2]
                t_ = matrix[r:r+3, c:c+3]
                t_[0][1] = np.linalg.norm(t[0][1]-t[0][0])
                t_[1][0] = np.linalg.norm(t[0][0]-t[1][0])
                t_[1][1] = (np.linalg.norm(t[0][0]-t[1][1]) + np.linalg.norm(t[0][1]-t[1][0]))/2
                t_[1][2] = np.linalg.norm(t[0][1]-t[1][1])
                t_[2][1] = np.linalg.norm(t[1][0]-t[1][1])

        for r in range(1,matrix.shape[0]-1,2):
            for c in range(1,matrix.shape[1]-1,2):
                slc = matrix[r-1:r+2, c-1:c+2]
                matrix[r][c] = np.mean(slc[slc != -1])

        if plot:
            ax = sns.heatmap(matrix[1:-1,1:-1], linewidth=0)#0.5, cmap="gray", square=True)
            plt.savefig("SOM_Umatrix.png",dpi=400)
        
        return matrix

if __name__ == "__main__":

    # For 3D Plotting of the SOM "in action".
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    #'''
    #=====================================================================================
    #======= Plane =======================================================================
    #=====================================================================================

    x = random.choices(np.arange(-1,1,0.01), k=500)
    y = random.choices(np.arange(-1,1,0.01), k=500)
    z = [i+j for i,j in zip(x,y)]

    plane = np.array([[i,j,k] for i,j,k in zip(x,y,z)])

    neighborhood_functions  = Neighborhood_Functions()
    learning_rate_functions = Learning_Rate_Functions()

    map = Planer_SOM(10,10,3, neighborhood_functions.cutgauss, learning_rate_functions.linear)
    print("Training SOM to plane dataset...")
    map.sequential_fit(plane, 2000)

    # map.umatrix()
    
    print("Generating SOM training plot for plane dataset...")
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    initial = True
    for h in map.history[::20]:
        ax.clear()
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.scatter(x,y,z,alpha=0.3, s=1)
        X, Y, Z = [],[],[]
        for i in h:
            for j in i:
                X.append([j[0]])
                Y.append([j[1]])
                Z.append([j[2]])
        ax.scatter(X,Y,Z, s=5)
        plt.draw()
        plt.pause(0.001)
        if initial:
            input()
            initial = False
    #'''
    #'''
    #=====================================================================================
    #======= Paraboloid ==================================================================
    #=====================================================================================

    x = random.choices(np.arange(-1,1,0.01), k=500)
    y = random.choices(np.arange(-1,1,0.01), k=500)
    z = [i**2+j**2 for i,j in zip(x,y)]

    para = np.array([[i,j,k] for i,j,k in zip(x,y,z)])

    neighborhood_functions  = Neighborhood_Functions(5, 500)
    learning_rate_functions = Learning_Rate_Functions(2, 2000)

    map = Planer_SOM(10,10,3, neighborhood_functions.gaussian, learning_rate_functions.power)
    print("Training SOM to paraboloid dataset...")
    map.sequential_fit(para, 2000)

    map.umatrix()
    print("Generating SOM training plot for paraboloid dataset...")
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    initial = True
    for h in map.history[::20]:
        ax.clear()
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.scatter(x,y,z,alpha=0.3, s=1)
        X, Y, Z = [],[],[]
        for i in h:
            for j in i:
                X.append([j[0]])
                Y.append([j[1]])
                Z.append([j[2]])
        ax.scatter(X,Y,Z, s=2)
        plt.draw()
        plt.pause(0.001)
        if initial:
            input()
            initial = False
    #'''
    #'''
    #=====================================================================================
    #======= Hyperboloid =================================================================
    #=====================================================================================


    x = random.choices(np.arange(-1,1,0.01), k=1000)
    y = random.choices(np.arange(-1,1,0.01), k=1000)
    z = [np.random.choice([-1,1])*np.sqrt((i**2 + j**2) + 0.25) for i,j in zip(x,y)]

    hyper = np.array([[i,j,k] for i,j,k in zip(x,y,z)])

    neighborhood_functions  = Neighborhood_Functions(sigma_t=5, T=500, end=2000)
    learning_rate_functions = Learning_Rate_Functions(2, 1000)

    map = Planer_SOM(20,20,3, neighborhood_functions.cutgauss, learning_rate_functions.linear)
    print("Training SOM to hyperboloid dataset...")
    map.sequential_fit(hyper, 2000)

    map.umatrix()

    print("Generating SOM training plot for hyperboloid dataset...")
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    initial = True
    for h in map.history[::20]:
        ax.clear()
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.scatter(x,y,z,alpha=0.3, s=1)
        X, Y, Z = [],[],[]
        for i in h:
            for j in i:
                X.append([j[0]])
                Y.append([j[1]])
                Z.append([j[2]])
        ax.scatter(X,Y,Z, s=2)
        plt.draw()
        plt.pause(0.0001)
        if initial:
            input()
            initial = False
    #'''
    #'''
    #=====================================================================================
    #======= Paraboloid Patches ==========================================================
    #=====================================================================================


    x = random.choices(np.arange(-1,-0.5,0.01), k=1000)
    x.extend(random.choices(np.arange(0.5,1,0.01), k=1000))
    y = random.choices(np.arange(-1,-0.5,0.01), k=1000)
    y.extend(random.choices(np.arange(0.5,1,0.01), k=1000))
    z = [(i**2+j**2)/4 for i,j in zip(x,y)]

    para = np.array([[i,j,k] for i,j,k in zip(x,y,z)])

    neighborhood_functions  = Neighborhood_Functions(5, 500)
    learning_rate_functions = Learning_Rate_Functions(2, 2000)

    map = Planer_SOM(10,10,3, neighborhood_functions.gaussian, learning_rate_functions.power)
    print("Training SOM to paraboloid patches dataset...")
    map.sequential_fit(para, 2000)
    

    map.umatrix()

    print("Generating SOM training plot for paraboloid patches dataset...")
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for h in map.history[::10]:
        ax.clear()
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.scatter(x,y,z,alpha=0.3, s=1)
        X, Y, Z = [],[],[]
        for i in h:
            for j in i:
                X.append([j[0]])
                Y.append([j[1]])
                Z.append([j[2]])
        ax.scatter(X,Y,Z, s=2)
        plt.draw()
        plt.pause(0.0001)
    #'''

    #'''
    #=====================================================================================
    #======= Fisher's Iris Subset ========================================================
    #=====================================================================================
    iris = np.array([    [5.1,3.5,1.4,0.2],
                [4.9,3.0,1.4,0.2],
                [4.7,3.2,1.3,0.2],
                [4.6,3.1,1.5,0.2],
                [5.0,3.6,1.4,0.2],
                [5.4,3.9,1.7,0.4],
                [4.6,3.4,1.4,0.3],
                [5.0,3.4,1.5,0.2],
                [4.4,2.9,1.4,0.2],
                [4.9,3.1,1.5,0.1],
                [5.4,3.7,1.5,0.2],
                [4.8,3.4,1.6,0.2],
                [4.8,3.0,1.4,0.1],
                [4.3,3.0,1.1,0.1],
                [5.8,4.0,1.2,0.2],
                [5.7,4.4,1.5,0.4],
                [5.4,3.9,1.3,0.4],
                [5.1,3.5,1.4,0.3],
                [5.7,3.8,1.7,0.3],
                [5.1,3.8,1.5,0.3],
                [5.4,3.4,1.7,0.2],
                [5.1,3.7,1.5,0.4],
                [4.6,3.6,1.0,0.2],
                [5.1,3.3,1.7,0.5],
                [4.8,3.4,1.9,0.2],
                [5.0,3.0,1.6,0.2],
                [5.0,3.4,1.6,0.4],
                [5.2,3.5,1.5,0.2],
                [5.2,3.4,1.4,0.2],
                [4.7,3.2,1.6,0.2],
                [4.8,3.1,1.6,0.2],
                [5.4,3.4,1.5,0.4],
                [5.2,4.1,1.5,0.1],
                [5.5,4.2,1.4,0.2],
                [4.9,3.1,1.5,0.1],
                [5.0,3.2,1.2,0.2],
                [5.5,3.5,1.3,0.2],
                [4.9,3.1,1.5,0.1],
                [4.4,3.0,1.3,0.2],
                [5.1,3.4,1.5,0.2],
                [5.0,3.5,1.3,0.3],
                [4.5,2.3,1.3,0.3],
                [4.4,3.2,1.3,0.2],
                [5.0,3.5,1.6,0.6],
                [5.1,3.8,1.9,0.4],
                [4.8,3.0,1.4,0.3],
                [5.1,3.8,1.6,0.2],
                [4.6,3.2,1.4,0.2],
                [5.3,3.7,1.5,0.2],
                [5.0,3.3,1.4,0.2],
                [7.0,3.2,4.7,1.4],
                [6.4,3.2,4.5,1.5],
                [6.9,3.1,4.9,1.5],
                [5.5,2.3,4.0,1.3],
                [6.5,2.8,4.6,1.5],
                [5.7,2.8,4.5,1.3],
                [6.3,3.3,4.7,1.6],
                [4.9,2.4,3.3,1.0],
                [6.6,2.9,4.6,1.3],
                [5.2,2.7,3.9,1.4],
                [5.0,2.0,3.5,1.0],
                [5.9,3.0,4.2,1.5],
                [6.0,2.2,4.0,1.0],
                [6.1,2.9,4.7,1.4],
                [5.6,2.9,3.6,1.3],
                [6.7,3.1,4.4,1.4],
                [5.6,3.0,4.5,1.5],
                [5.8,2.7,4.1,1.0],
                [6.2,2.2,4.5,1.5],
                [5.6,2.5,3.9,1.1],
                [5.9,3.2,4.8,1.8],
                [6.1,2.8,4.0,1.3],
                [6.3,2.5,4.9,1.5],
                [6.1,2.8,4.7,1.2],
                [6.4,2.9,4.3,1.3],
                [6.6,3.0,4.4,1.4],
                [6.8,2.8,4.8,1.4],
                [6.7,3.0,5.0,1.7],
                [6.0,2.9,4.5,1.5],
                [5.7,2.6,3.5,1.0],
                [5.5,2.4,3.8,1.1],
                [5.5,2.4,3.7,1.0],
                [5.8,2.7,3.9,1.2],
                [6.0,2.7,5.1,1.6],
                [5.4,3.0,4.5,1.5],
                [6.0,3.4,4.5,1.6],
                [6.7,3.1,4.7,1.5],
                [6.3,2.3,4.4,1.3],
                [5.6,3.0,4.1,1.3],
                [5.5,2.5,4.0,1.3],
                [5.5,2.6,4.4,1.2],
                [6.1,3.0,4.6,1.4],
                [5.8,2.6,4.0,1.2],
                [5.0,2.3,3.3,1.0],
                [5.6,2.7,4.2,1.3],
                [5.7,3.0,4.2,1.2],
                [5.7,2.9,4.2,1.3],
                [6.2,2.9,4.3,1.3],
                [5.1,2.5,3.0,1.1],
                [5.7,2.8,4.1,1.3],
                [6.3,3.3,6.0,2.5],
                [5.8,2.7,5.1,1.9],
                [7.1,3.0,5.9,2.1],
                [6.3,2.9,5.6,1.8],
                [6.5,3.0,5.8,2.2],
                [7.6,3.0,6.6,2.1],
                [4.9,2.5,4.5,1.7],
                [7.3,2.9,6.3,1.8],
                [6.7,2.5,5.8,1.8],
                [7.2,3.6,6.1,2.5],
                [6.5,3.2,5.1,2.0],
                [6.4,2.7,5.3,1.9],
                [6.8,3.0,5.5,2.1],
                [5.7,2.5,5.0,2.0],
                [5.8,2.8,5.1,2.4],
                [6.4,3.2,5.3,2.3],
                [6.5,3.0,5.5,1.8],
                [7.7,3.8,6.7,2.2],
                [7.7,2.6,6.9,2.3],
                [6.0,2.2,5.0,1.5],
                [6.9,3.2,5.7,2.3],
                [5.6,2.8,4.9,2.0],
                [7.7,2.8,6.7,2.0],
                [6.3,2.7,4.9,1.8],
                [6.7,3.3,5.7,2.1],
                [7.2,3.2,6.0,1.8],
                [6.2,2.8,4.8,1.8],
                [6.1,3.0,4.9,1.8],
                [6.4,2.8,5.6,2.1],
                [7.2,3.0,5.8,1.6],
                [7.4,2.8,6.1,1.9],
                [7.9,3.8,6.4,2.0],
                [6.4,2.8,5.6,2.2],
                [6.3,2.8,5.1,1.5],
                [6.1,2.6,5.6,1.4],
                [7.7,3.0,6.1,2.3],
                [6.3,3.4,5.6,2.4],
                [6.4,3.1,5.5,1.8],
                [6.0,3.0,4.8,1.8],
                [6.9,3.1,5.4,2.1],
                [6.7,3.1,5.6,2.4],
                [6.9,3.1,5.1,2.3],
                [5.8,2.7,5.1,1.9],
                [6.8,3.2,5.9,2.3],
                [6.7,3.3,5.7,2.5],
                [6.7,3.0,5.2,2.3],
                [6.3,2.5,5.0,1.9],
                [6.5,3.0,5.2,2.0],
                [6.2,3.4,5.4,2.3],
                [5.9,3.0,5.1,1.8]
            ])

    x = [i[0] for i in iris]
    y = [i[1] for i in iris]
    z = [i[2] for i in iris]
    w = [i[3] for i in iris]

    neighborhood_functions  = Neighborhood_Functions(5, 500)
    learning_rate_functions = Learning_Rate_Functions(4, 2000)

    map = Planer_SOM(20,20,4, neighborhood_functions.gaussian, learning_rate_functions.power)
    print("Training SOM to Fisher's iris dataset...")
    map.sequential_fit(iris, 1000)

    map.umatrix()

    print("Generating SOM training plot for Fisher's iris dataset...")
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    storage = []
    for h in map.history:
        ax.clear()
        ax.set_xlim(np.min(x)-0.5,np.max(x)+0.5)
        ax.set_ylim(np.min(y)-0.5,np.max(y)+0.5)
        ax.set_zlim(np.min(z)-0.5,np.max(z)+0.5)
        ax.scatter(x,y,z,alpha=0.3, s=1)
        X, Y, Z = [],[],[]
        for i in h:
            for j in i:
                X.append([j[0]])
                Y.append([j[1]])
                Z.append([j[2]])
        ax.scatter(X,Y,Z, s=2)
        plt.draw()
        plt.pause(0.001)
    with open("fisher.pickle","wb") as fout:
        pickle.dump(storage, fout)
    #'''

plt.show()
input()