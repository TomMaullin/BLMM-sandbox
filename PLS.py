import numpy as np
import cvxopt
from cvxopt import cholmod, umfpack, amd
import time
import pandas as pd

# Mapping function
# ----------------------------------------------
# This function takes in a vector of parameters,
# theta, and maps them the to lower triangular 
# block diagonal matrix, lambda.
# ----------------------------------------------
# The following inputs are required for this
# function:
#
# - theta: the vector of theta parameters
# - nlevels: a vector of the number of levels
#            for each grouping factor. e.g.
#            nlevels=[10,2] means there are 
#            10 levels for factor 1 and 2
#            levels for factor 2.
# - nparams: a vector of the number of 
#            variables for each grouping factor.
#            e.g. nlevels=[3,4] means there  
#            are 3 variables for factor 1 and 4
#            variables for factor 2.
#
# All arrays must be np arrays.
def mapping(theta, nlevels, nparams):

    # Work out how many factors there are
    n_f = len(nlevels)

    # Quick check that nlevels and nparams are the same length
    if len(nlevels)!=len(nparams):
        raise Exception('The number of parameters and number of levels should be recorded for every grouping factor.')

    # Work out how many lambda components needed for each factor
    n_lamcomps = (np.multiply(nparams,(nparams+1))/2).astype(np.int64)

    # Block index is the index of the next un-indexed diagonal element
    # of Lambda
    block_index = 0

    # Row indices and column indices of theta
    row_indices = np.array([])
    col_indices = np.array([])

    # This will have the values of theta repeated several times, once
    # for each time each value of theta appears in lambda
    theta_repeated = np.array([])
    
    # Loop through factors generating the indices to map theta to.
    for i in range(0,n_f):

        # Work out the indices of a lower triangular matrix
        # of size #variables(factor) by #variables(factor)
        row_inds_tri, col_inds_tri = np.tril_indices(nparams[i])

        # Work out theta for this block
        theta_current = theta[np.sum(n_lamcomps[0:i]):np.sum(n_lamcomps[0:(i+1)])]

        # For each level of the factor we must repeat the lower
        # triangular matrix
        for j in range(0,nlevels[i]):

            # Append the row/column indices to the running list
            row_indices = np.hstack((row_indices, (row_inds_tri+block_index)))
            col_indices = np.hstack((col_indices, (col_inds_tri+block_index)))

            # Repeat theta for every block
            theta_repeated = np.hstack((theta_repeated, theta_current))

            # Move onto the next block
            block_index = block_index + nparams[i]
            
    # Create lambda as a sparse matrix
    lambda_theta = cvxopt.spmatrix(theta_repeated.tolist(), row_indices.astype(np.int64), col_indices.astype(np.int64))

    # Return lambda
    return(lambda_theta)

# Inverse mapping function
# ----------------------------------------------
# This function takes in a lower triangular 
# block diagonal matrix, lambda, and maps it to
# the original vector of parameters, theta.
# ----------------------------------------------
# The following inputs are required for this
# function:
#
# - Lambda: The sparse lower triangular block
#           diagonal matrix. 
def inv_mapping(Lambda):

    # List the unique elements of lambda (in the
    # correct order; pandas does this, numpy does
    # not)
    theta = pd.unique(list(cvxopt.spmatrix.trans(Lambda)))
    return(theta)

# Examples
nparams = np.array([9,6,12,3,2,1])
nlevels = np.array([10,3,9,3,2,6])
theta = np.random.randn((np.sum(np.multiply(nparams,(nparams+1))/2)).astype(np.int64))
l=mapping(theta, nlevels, nparams)
print(inv_mapping(l)==theta)
