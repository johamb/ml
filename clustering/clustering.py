import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# example function definition
def testFunction(data):
    print(data)
    datasum = np.sum(data)
    return datasum

def k_means(k, distance):
    return

def L1(i, j, data):
    return

# main program code

# read data from file
data_pd = pd.read_csv('data.txt', header=None, sep='\s+')

# convert pandas to numpy array
# access elements as data[row,col]
data = data_pd.values

# example for function call
sumdata = testFunction(data)
print("Sum: ", sumdata)

# plot data as scatter plot
plt.plot(data[:,0],data[:,1],'b+')
plt.show()
