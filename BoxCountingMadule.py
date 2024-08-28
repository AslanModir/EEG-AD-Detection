#!/usr/bin/env python
# coding: utf-8

# In[1]:


def BoxCounting(Trajectories,K):
    count = 0
    N       = np.zeros(K)
    S       = np.zeros(K)
    MinXBox = min(Trajectories[0])
    MaxXBox = max(Trajectories[0])
    MinYBox = min(Trajectories[1])
    MaxYBox = max(Trajectories[1])
    dX = MaxXBox - MinXBox
    dY = MaxYBox - MinYBox
    if dX > dY:
        MainBoxSize = dX
    else:
        MainBoxSize = dY
    MainBoxPosition = np.matrix([[MinXBox, MinXBox + MainBoxSize, MinXBox, MinXBox + MainBoxSize],\
                                 [MinYBox, MinYBox, MinYBox + MainBoxSize, MinYBox + MainBoxSize]])
    #plt.plot(MainBoxPosition[0],MainBoxPosition[1],'*')
    for k in range(K):
        NewDot = (MinXBox, MinYBox)
        BoxNumSide = (2 ** k)                 # The number of boxes for each side of the box
        BoxSize    = MainBoxSize / BoxNumSide # The length of each side of the box
        for boxnumX in range(BoxNumSide): 
            for boxnumY in range(BoxNumSide): 
                BoxPosition = np.matrix([[NewDot[0], NewDot[0] + BoxSize, NewDot[0], NewDot[0] + BoxSize],\
                                         [NewDot[1], NewDot[1], NewDot[1] + BoxSize, NewDot[1] + BoxSize]])
                plt.plot(BoxPosition[0],BoxPosition[1],'*') #plt.show()
                for i in range(len(Trajectories[0])):
                    if (BoxPosition[0,0] < Trajectories[0][i] < BoxPosition[0,1]) &\
                       (BoxPosition[1,0] < Trajectories[1][i] < BoxPosition[1,2]):
                        count += 1 
                        break 
                NewDot = (NewDot[0] + BoxSize, NewDot[1])
            NewDot = (MinXBox, MinXBox + ((boxnumX + 1) * BoxSize))
        N[k]  = count
        S[k]  = 2 ** (K-(k+1)) + 0.0
        count = 0
    #S = S[1::]; N = N[1::] # The main box is ommited to increase the estimation percise since it is fix for all
    print("The number of counted box series (N) = " + str(N))
    print("The scale of box (S) = " + str(S))
    slope, interception = np.polyfit(np.log10(1/S), np.log10(N), 1) #find line of best fit by Least-mean-square-error (LMSE) technique
    '''
    plt.show()
    plt.scatter(np.log10(1/S), np.log10(N), color='purple')
    plt.plot(np.log10(1/S), slope*np.log10(1/S)+interception, color='steelblue', linestyle='--', linewidth=2)
    plt.xlabel ('log (1/S)', size=14) # For above example: 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1
    plt.ylabel ('log (N)', size=14)   # For above example: 1   , 4   , 15  , 50 , 152, 423, 928
    plt.text(np.min(np.log10(1/S)), np.max(np.log10(N)), 'y = ' + '{:.2f}'.format(slope) + 'x' + ' + ' + '{:.2f}'.format(interception), size=14)
    '''
    return slope


# In[2]:


import scipy
from scipy.io import loadmat # To read Mat files
import mne
import matplotlib.pyplot as plt
import numpy as np
import timeit
import BoxCountingMadule

Trajectories = (np.rand(1600) ,np.rand(1600))
K = 3

BoxCountingMadule.BoxCounting(Trajectories,K)


# In[ ]:




