#!/usr/bin/env python
# coding: utf-8

# Assignment 2 part 1 made by:
# Niels van den Hork - s4572602
# Niels van Drueten - s4496604

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

# PRESENT Cipher SBox
SBox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
# Inverse PRESENT Cipher SBox
SBox_inv = [0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA]

# Function f is the intermediate result,
# where i is the known non-constant data value
# and k is a small part of the key.
def f(i, k):
    i1, i2 = i >> 4, i & 0x0F
    k1, k2 = k >> 4, k & 0x0F
    inv1, inv2 = SBox[i1] ^ k1, SBox[i2] ^ k2
    result = inv1 << 4 ^ inv2
    return result

# Function f_inv is the inverse of function f,
# where o is the known non-constant data value
# and k is a small part of the key.
def f_inv(o, k):
    o1, o2 = o >> 4, o & 0x0F
    k1, k2 = k >> 4, k & 0x0F
    inv1, inv2 = SBox_inv[o1] ^ k1, SBox_inv[o2] ^ k2
    result = inv1 << 4 ^ inv2
    return result

# Returns the Hamming distance between val1 and val2.
def hd(val1,val2):
    return bin(val1 ^ val2).count("1")

# INPUT MATRIX WILL BE OF SIZE [no_inputs x no_keys]
# INSTEAD OF [no_inputs x 1]
def construct_input_matrix(output, key_len):
	input = np.zeros((len(output), 2**key_len), dtype="uint8")
	for i in range(len(output)):
		output_elem = output[i][0]
		for k in range(2**key_len):
			val = f_inv(output_elem,k)
			input[i][k] = val
	return input

# Returns a Power-Prediction Matrix of size [no_inputs x no_keys]
# Input _in: Value-Prediction Matrix of size [no_inputs x no_keys]
def construct_pow_pred_matrix(val_pred_matrix, od, key_len):
	output = np.zeros((len(_in), 2**key_len), dtype="uint8")
	for i in range(len(_in)):
		in_elem = _in[i][0]
		for k in range(2**key_len):
			val1 = val_pred_matrix[i][k]
			val2 = od[i][0]
			output[i][k] = hd(val1, val2)
	return output

# Uses the correlate function of the scipy io library,
# that cross-correlates two matrices.
def correlate_m(matrix1, matrix2):
    print("Correlating between: {} and {}".format(matrix1.shape,matrix2.shape))
    
    cols_matrix1 = matrix1.shape[1]
    cols_matrix2 = matrix2.shape[1]
    
    result = np.zeros((cols_matrix1,cols_matrix2))
    
    for j in tqdm(range(cols_matrix2)):
        for i in range(cols_matrix1):
            #result[i][j] = pearsonr(matrix1[:,i], matrix2[:,j])[0]
            result[i][j] = np.corrcoef(matrix1[:,i], matrix2[:,j])[0][1]
    return result

# Computes best key based on absolute correlation value.
def compute_best_key(result):
    key = 0
    value = 0
    for i in range(result.shape[0]):
        this_value = 0
        for j in range(result.shape[1]):
            this_value = abs(result[i][j])
            #print("key: {}, {} difference {}".format(i, this_value, value - this_value))
            if(this_value > value):
                #print("old value: {} with key: {}".format(value, i))
                value = this_value
                key = i
                print("new value: {} with key: {} (change: {})".format(value, i, this_value - value))
        
    return key, value

# Opens "hardware_traces.mat" file.
ht_file = loadmat('hardware_traces.mat')
ht = ht_file['traces'] #contains 10.000 traces, 2k samples each, 8bit(?) values

# Opens "output_data.mat" file.
od_file = loadmat('output_data.mat')
od = od_file['output_data'] #contains 10.000 output values, each 8bit

# Reconstruct an input matrix from the output matrix
_in = construct_input_matrix(od, 8)
print("Reconstructed input prediction matrix: \n {} {}".format(_in, _in.shape))
# We do not have to compute an value-prediction matrix as we already
# know the output.

# Computing power prediction matrix
pow_pred_matrix = construct_pow_pred_matrix(_in, od, 8)
print("Power prediction matrix: \n {} {}".format(pow_pred_matrix, pow_pred_matrix.shape))

# Opens "traces.mat" file.
trace_file = loadmat('hardware_traces.mat')
_traces = trace_file['traces']
print("Traces matrix: \n {} {}".format(_traces, _traces.shape ))

result = correlate_m(pow_pred_matrix, _traces)
print(result.shape)


best_keyguess, value = compute_best_key(result)
print("Best key: {} with absolute value: {}".format(best_keyguess, value))

plt.plot([sum(list(map(abs,row))) for row in result])
plt.title('Key Candidates based on absolute correlation')
plt.xlabel('key')
plt.ylabel('summed absolute correlation')
plt.show()


absresult = np.array([list(map(abs,row)) for row in result])
maxidx = np.argmax(absresult,axis=1)

maxconf = np.array([(row[0],midx,row[1][midx]) for row,midx in zip(enumerate(absresult),maxidx)])
smaxconf = np.array(sorted(maxconf,key = lambda x : -x[2]) )

#[print(e) for e in smaxconf]
    
plt.bar(range(256),maxconf[:,2] )
plt.title('Key Candidates based on absolute correlation')
plt.xlabel('key')
plt.ylabel('peak absolute correlation')
plt.show()


for i,row in enumerate(result):
    if i == best_keyguess:
        continue
    plt.plot(list(map(abs,row)),color='gray')
    
plt.plot(list(map(abs,result[best_keyguess])),color='blue')
plt.title('Absolute correlation of every key candidate (blue = {})'.format(best_keyguess))
plt.xlabel('time samples')
plt.ylabel('absolute correlation')
plt.show()

keyranking = []
for amount in [500,1000,2000,4000,8000,12000]:
    result = correlate_m(pow_pred_matrix[:amount], _traces[:amount])

    absresult = np.array([list(map(abs,row)) for row in result])
    maxidx = np.argmax(absresult,axis=1)
    maxconf = np.array([(row[0],midx,row[1][midx]) for row,midx in zip(enumerate(absresult),maxidx)])
    smaxconf = np.array(sorted(maxconf,key = lambda x : -x[2]) )
    #[print(e) for e in smaxconf]

    keyrank = np.array([e[0] for e in smaxconf])
    print("Current best guess: {}".format(keyrank[0]))
    keyidx = np.where(keyrank == best_keyguess)[0][0]
    print("Ranking of {}: {}".format(best_keyguess, keyidx))
    keyranking.append(keyidx)

plt.plot(np.array([500,1000,2000,4000,8000,12000]),keyranking)  
plt.title('Ranking of key={}'.format(best_keyguess)) 
plt.xlabel('amount of traces')
plt.ylabel('ranking (lower is better)')
plt.show()

