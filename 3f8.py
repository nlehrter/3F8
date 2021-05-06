import numpy as np
import code as c
import matplotlib.pyplot as plt
import random

import matplotlib.pyplot as plt

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

def plot_data_internal(X, y):
    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), \
        np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1',fontsize=30)
    plt.ylabel('X2',fontsize = 30)
    plt.title('Plot data',fontsize = 30)
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    plt.rcParams.update({'font.size':30})
    plt.tick_params(labelsize = 20)
    return xx, yy

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

def plot_data(X, y):
    xx, yy = plot_data_internal(X, y)
    plt.show()

##
# x: input to the logistic function
#

def logistic(x): return 1.0 / (1.0 + np.exp(-x))

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# w: current parameter values
#

def compute_average_ll(X, y, w):
    output_prob = logistic(np.dot(X, w))
    return np.mean(y * np.log(output_prob.T) + (1 - y) * np.log(1.0 - output_prob.T))

##
# ll: 1d array with the average likelihood per data point and dimension equal
#     to the number of training epochs.
#

def plot_ll(ll):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps',fontsize = 30)
    plt.ylabel('Average log-likelihood',fontsize = 30)
    plt.title('Plot Average Log-likelihood Curve',fontsize = 30)
    plt.tick_params(labelsize = 20)
    plt.show()

##
# x: 2d array with input features at which to compute predictions.
#
# (uses parameter vector w which is defined outside the function's scope)
#

def predict_for_plot(x): 
    x_tilde = np.concatenate((np.ones((x.shape[ 0 ], 1 )), x), 1)
    return logistic(np.dot(x_tilde, w))

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# predict: function that recives as input a feature matrix and returns a 1d
#          vector with the probability of class 1.

def plot_predictive_distribution(X, y, predict):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_predict = np.concatenate((xx.ravel().reshape((-1, 1)), \
        yy.ravel().reshape((-1, 1))), 1)
    Z = predict(X_predict)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.tick_params(labelsize = 20)
    plt.show()

##
# l: hyper-parameter for the width of the Gaussian basis functions
# Z: location of the Gaussian basis functions
# X: points at which to evaluate the basis functions

def expand_inputs(l, X, Z):
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[ 0 ])
    ones_X = np.ones(X.shape[ 0 ])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)

##
# x: 2d array with input features at which to compute the predictions
#    using the feature expansion
#
# (uses parameter vector w and the 2d array X with the centers of the basis
# functions for the feature expansion, which are defined outside the function's
# scope)
#

def predict_for_plot_expanded_features(x):
    x_expanded = expand_inputs(l, x, X)
    x_tilde = np.concatenate((np.ones((x_expanded.shape[ 0 ], 1 )), x_expanded), 1)
    return logistic(np.dot(x_tilde, w))


def generate_cost_matrix(X_test,w):
    y_outcome = logistic(X_test.dot(w))
    y_fin = abs(y_outcome.astype(int))
    fail0 = 0
    fail1 = 0
    ones = 0
    for i in range(0,len(y_test)):
        if y_outcome[i] <0.5:
            y_fin[i] =0
        else:
            y_fin[i] = 1
        
        if y_test[i] == 1:
            ones += 1
            if y_fin[i] == 0:
                fail1 += 1 
        elif y_test[i] == 0:
            if y_fin[i] == 1:
                fail0 += 1 
    
    mat = np.array([[200-ones-fail0,fail0],[fail1,ones-fail1]])/200
    return mat
"""
def ascent(X,y,w,iterations, eta):
    w_history = []
    log_likelihood_history = []

    for i in range(0,iterations):
        w_history.append(w)
        logis = logistic(X.dot(w))
        log_lik = y.dot(np.log(logis)) + (1-y).dot(np.log(1-logis))
        
        log_likelihood_history.append(compute_average_ll(X,y,w))
        if i<100:
            if log_likelihood_history[i] > -0.72:
                print(i)
                print(log_lik/800)
                
        dl = (y - np.transpose(logis)).dot(X)

        w = w + eta* np.transpose(dl)

    return w,w_history,log_likelihood_history
"""
def ascent(X,y,w,iterations, eta):
    w_history = []
    log_likelihood_history = []
    test_ll = []
    
    for i in range(0,iterations):
        w_history.append(w)
        logis = logistic(X.dot(w))
        dl = (y - np.transpose(logis)).dot(X)
        log_likelihood_history.append(compute_average_ll(X,y.T,w))
        test_ll.append(compute_average_ll(X_test,y_test.T,w))
        w = w + eta* np.transpose(dl)

    return w,w_history,log_likelihood_history, test_ll


"""Studying the data it seems that a linear classifier will struggle with this dataset as  one class is broken
up into several distinct sets which surround the other set. However, the regions are clearly defined and so it 
may be able to define"""
X_prov = np.loadtxt('X.txt')
y_prov = np.loadtxt('y.txt')
z= np.arange(0,np.shape(y_prov)[0],1)

random.shuffle(z)
y = np.zeros(1000)
X = np.zeros((1000,2))
for i in range(0,np.shape(y_prov)[0]):
    y[i] = y_prov[z[i]]
    X[i] = X_prov[z[i]]

plot_data(X,y)
a = np.ones(1000)
X_p = X[:800]

X2=X
X = np.hstack((a.reshape((1000, 1)),X))

y_p = y

X_train = X[:800]
X_test = X[800:]
y_train = y[:800]
y_test = y[800:]

w_initial = np.array([[0.0],[0.0],[0.0]])
w,w_history,log_likelihood_history, test_ll = ascent(X_train,y_train,w_initial,50,0.001)
print(log_likelihood_history[-1])
print(test_ll[-1])

print("Results for the linear model were found to be:")
print("The average log-likelihood:")
print("Confusion matrix:")
print(generate_cost_matrix(X_test,w))
plot_ll(log_likelihood_history)
plot_ll(test_ll)
plot_predictive_distribution(X2,y,predict_for_plot)



#print(test_ll)