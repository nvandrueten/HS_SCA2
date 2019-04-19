#!/usr/bin/env python
# coding: utf-8

# Assignment 2 part 2 made by:

# Niels van den Hork - s457260
# Niels van Drueten - s44966044


from scipy.io import loadmat
from scipy import signal
import numpy as np
import json
import matplotlib.pyplot as plt
#!pip install tqdm
#from tqdm import tqdm_notebook as tqdm #if running in a notebook
from tqdm import tqdm as tqdm #if not running in a notebook
from scipy.stats.stats import pearsonr
#from numpy import correlate as corr #not pearson 
from pylab import rcParams


KEYSIZE = 4
INPUTSIZE = 4
RANDSIZE = 4
SBox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


def correlate_m(matrix1, matrix2):
    print(matrix1.shape,matrix2.shape)
    
    cols_matrix1 = matrix1.shape[1]
    cols_matrix2 = matrix2.shape[1]
    
    result = np.zeros((cols_matrix1,cols_matrix2))
    
    for j in tqdm(range(cols_matrix2)):
        for i in range(cols_matrix1):
            #result[i][j] = pearsonr(matrix1[:,i], matrix2[:,j])[0]
            result[i][j] = np.corrcoef(matrix1[:,i], matrix2[:,j])[0][1]
    return result



# Opens "hardware_traces.mat" file.
file = loadmat('input.mat')
inp = file['input'] #contains 2000 4-bit nipples

# Opens "hardware_traces.mat" file.
file = loadmat('leakage_y0_y1.mat')
leaks = file['L'] #contains 2.000 traces, 10 samples each, 4bit nipples

print("Input: \n {} {}".format(leaks, leaks.shape))

traces = np.zeros((2000,45)) # 10 choose 2 = 45
for idx0,trace in enumerate(leaks):
    count = 0
    for idx1,val1 in enumerate(trace):
        for idx2,val2 in enumerate(trace[idx1+1:]):
            traces[idx0][count] = val1 * val2
            count += 1

value_pred = np.zeros((inp.shape[0],2**KEYSIZE))
for idx1, in_ in enumerate(inp[:,0]):
    for key in range(2**KEYSIZE):
        value_pred[idx1,key] = SBox[in_^key]
        
print(value_pred[0],value_pred.shape)
plt.plot(value_pred[0])
plt.title('value prediction for input 0')
plt.xlabel('Key')
plt.ylabel('value')
plt.show()

correlations = correlate_m(traces,value_pred)
print(correlations.shape)


rcParams['figure.figsize'] = 25, 25


fig = plt.figure()
fig.tight_layout()
main = fig.add_subplot(111,frame_on = False)

for i in range(correlations.shape[1]):
    abs_cor = np.array([abs(c) for c in correlations])
    
    ax = fig.add_subplot(4,4,i+1)
    ax.set_title('key '+str(i))
    ax.plot(abs_cor[:,:],color='gray',alpha=0.5)
    ax.plot(abs_cor[:,i],color='red')

main.set_xlabel('possible multiplications',fontsize=26,labelpad=20)
main.set_ylabel('correlation',fontsize=26,labelpad=20)
main.set_title('Absolute Correlation between Traces and Value Prediction',fontsize=26,pad=30)
plt.show()


# From these plots we can clearly see that key 3 has the highest correlation
