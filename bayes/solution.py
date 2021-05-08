import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd

def error_ellipse(covmat, meanvec, ax, n_std = 2):
    """
    Create a plot of the covariance confidence ellipse

    Parameters
    ----------
    covmat, meanvec: covariance matrix (2x2) and mean vector (2x1) of data
    ax: plot axes; get with ax = plt.gca()
    n_std: number of standard deviations for radius of ellipse

    based on plot_confidence_ellipse.py by Carsten Schelp (https://github.com/CarstenSchelp/CarstenSchelp.github.io)
    """
    pearson = covmat[0, 1] / np.sqrt(covmat[0, 0] * covmat[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, fill=False)

    # calculating the standard deviation of x from  the square root of the variance
    scale_x = np.sqrt(covmat[0, 0]) * n_std
    mean_x = meanvec[0]

    # calculating the standard deviation of y from  the square root of the variance
    scale_y = np.sqrt(covmat[1, 1]) * n_std
    mean_y = meanvec[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    return


# main program code

# read data from file
dataX_pd = pd.read_csv('x.dat', header=None, sep='\s+')
dataY_pd = pd.read_csv('y.dat', header=None, sep='\s+')

# convert pandas to numpy array
# access elements as data[row,col]
x = dataX_pd.values
y = dataY_pd.values

# extract pos/neg
# returns the indices of the rows meeting the specified condition
pos = np.where(y == 1)
pos = pos[0]
neg = np.where(y == 0)
neg = neg[0]

# plot data as scatter plot
plt.plot(x[pos,0], x[pos,1],'b+')
plt.plot(x[neg,0], x[neg,1],'bo')


# training
# class priors
pPos = np.size(pos, 0) / np.size(y, 0)
pNeg = np.size(neg, 0) / np.size(y, 0)

# determine parameters of normal distribution
covpos = np.cov(np.transpose(x[pos,:]))
covneg = np.cov(np.transpose(x[neg,:]))

mupos = np.mean(x[pos,:],0)
muneg = np.mean(x[neg,:],0)

# plot error ellipses
ax = plt.gca()
error_ellipse(covpos, mupos, ax)
error_ellipse(covneg, muneg, ax)
plt.show()

# classify all vectors (compute posterior probability for each class)
covposinv = np.linalg.inv(covpos)
covneginv = np.linalg.inv(covneg)

factorPos = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covpos)))
factorNeg = 1 / (np.sqrt(np.linalg.det(2 * np.pi * covneg)))

posteriorsPos = np.zeros(np.size(y, 0))
posteriorsNeg = np.zeros(np.size(y, 0))

for i in range(0, np.size(y, 0)-1):
    # positive class
    diffToCenter = x[i,:] - mupos
    exponent = -0.5 * np.matmul(diffToCenter, np.matmul(covposinv, diffToCenter))

    posteriorsPos[i] = pPos * factorPos * np.exp(exponent)
    # negative class
    diffToCenter = x[i, :] - muneg
    exponent = -0.5 * np.matmul(diffToCenter, np.matmul(covneginv, diffToCenter))

    posteriorsNeg[i] = pNeg * factorNeg * np.exp(exponent)

    # p(c)
    pC = posteriorsPos[i] + posteriorsNeg[i]
    # normalization
    posteriorsPos[i] = posteriorsPos[i] / pC
    posteriorsNeg[i] = posteriorsNeg[i] / pC

# decision (for class with highest probability) and error rate on training set
decision = np.zeros(np.size(y, 0))

for i in range(0, np.size(y, 0)-1):
    if posteriorsPos[i] > posteriorsNeg[i]:
        decision[i] = 1
    else:
        decision[i] = 0

# error rate
error = abs(decision - y[:,0])
noIncorrect = np.sum(error)
errorRate = noIncorrect / np.size(error)
print('Correctly classified:', (1 - errorRate) * 100, '%')
print('Error rate:', errorRate * 100, '%')

incorrect = np.where(error == 1)
print('Number of incorrectly classified samples:', np.size(incorrect))

# plot data as scatter plot
plt.plot(x[pos,0], x[pos,1],'b+')
plt.plot(x[neg,0], x[neg,1],'bo')

# plot error ellipses
ax = plt.gca()
error_ellipse(covpos, mupos, ax)
error_ellipse(covneg, muneg, ax)

plt.plot(x[incorrect, 0], x[incorrect, 1], 's')
plt.show()
