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
# compute class priors

# determine parameters of normal distribution

# mean vector: np.mean (caution: provide axis argument)
mupos = np.mean(x[pos], axis=0)
print(mupos)
muneg = np.mean(x[neg], axis=0)
print(muneg)

# covariance matrix: np.cov
covpos = np.cov(x[pos], rowvar=False)
covneg = np.cov(x[neg], rowvar=False)
print(covpos)

# plot error ellipses
ax = plt.gca()
error_ellipse(covpos, mupos, ax)
error_ellipse(covneg, muneg, ax)
plt.show()


# classify all vectors (compute posterior probability for each class)
# matrix inversion: np.linalg.inv
# matrix multiplication: np.matmul
def compute_probability(feature_vector, mean_vector, covariance_matrix):
    factor = 1 / np.sqrt(np.linalg.det(2 * np.pi * covariance_matrix))
    inverse = np.linalg.inv(covariance_matrix)
    diff = feature_vector - mean_vector
    tmp = -0.5 * diff
    tmp2 = np.dot(tmp, inverse)
    exp = np.dot(tmp2, diff)
    exponential = np.exp(exp)
    return factor * exponential


probabilities = np.empty([len(x), 2])

i = 0
for feature_vector in x:
    prob = compute_probability(feature_vector, mupos, covpos)
    probabilities[i, 0] = prob
    i += 1

i = 0
for feature_vector in x:
    probabilities[i, 1] = compute_probability(feature_vector, muneg, covneg)
    i += 1

# decision (for class with highest probability) and error rate on training set
i = 0
failures = 0
for p in probabilities:
    if p[0] > p[1]:
        print('{}: approved'.format(i))
        if y[i] != 1:
            failures += 1
    else:
        print('{}: failed'.format(i))
        if y[i] != 0:
            failures += 1
    i += 1

# compute error rate
print('Error rate: {}'.format(failures/80))
