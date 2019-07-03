import numpy as np
import cvxopt
from cvxopt import cholmod, umfpack, amd, matrix, spmatrix
import time
import pandas as pd
###
import os

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
    lambda_theta = spmatrix(theta_repeated.tolist(), row_indices.astype(np.int64), col_indices.astype(np.int64))

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

# Sparse Cholesky Decomposition function
# ----------------------------------------------
# This function takes in a square matrix M and
# outputs P and L from it's sparse cholesky
# decomposition of the form PAP'=LL'.
#
# Note: P is given as a permutation vector
# rather than a matrix.
# ----------------------------------------------
# The following inputs are required for this
# function:
#
# - M: The matrix to be sparse cholesky
#      decomposed as an spmatrix from the
#      cvxopt package.
def sparse_chol(M):

    # Quick check that M is square
    if M.size[0]!=M.size[1]:
        raise Exception('M must be square.')

    # Set the factorisation to use LL' instead of LDL'
    cholmod.options['supernodal']=2

    # Make an expression for the factorisation
    F=cholmod.symbolic(M)

    # Calculate the factorisation
    cholmod.numeric(M, F) 

    # Set p to [0,...,n-1]
    P = cvxopt.matrix(range(M.size[0]), (M.size[0],1), tc='d')

    # Solve and replace p with the true permutation used
    cholmod.solve(F, P, sys=7)

    # Convert p into an integer array; more useful that way
    P=cvxopt.matrix(np.array(P).astype(np.int64),tc='i')

    # Get the sparse cholesky factor
    L=cholmod.getfactor(F)

    # Return P and L
    return(P,L)
    

    

# Examples
nparams = np.array([9,6,12,3,2,1])
nlevels = np.array([10,3,9,3,2,6])
theta = np.random.randn((np.sum(np.multiply(nparams,(nparams+1))/2)).astype(np.int64))
l=mapping(theta, nlevels, nparams)
print(inv_mapping(l)==theta)

# Go to test data directory
os.chdir('/home/tommaullin/Documents/BLMM-sandbox/testdata')

# Read in random effects variances
D_3col=pd.read_csv('true_rfxvar_3col.csv',header=None).values
D = cvxopt.spmatrix(D_3col[:,2].tolist(), (D_3col[:,0]-1).astype(np.int64), (D_3col[:,1]-1).astype(np.int64))

P,L = sparse_chol(D)
LLt=L*cvxopt.spmatrix.trans(L)
print(LLt-D[P,P])
