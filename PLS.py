import numpy as np
import cvxopt
from cvxopt import cholmod, umfpack, amd, matrix, spmatrix, lapack
import pandas as pd
###
import os
import time
import numba

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

# Sparse symmetric determinant
# ----------------------------------------------
# This function takes in a square symmetric
# matrix M and outputs it's determinant
# ----------------------------------------------
# The following inputs are required for this
# function:
#
# - M: A sparse symmetric matrix
#
# Note: THIS DOES NOT WORK FOR NON-SYMMETRIC M
def sp_det_sym(M):

    # Change to the LL' decomposition
    prevopts = cholmod.options['supernodal']
    cholmod.options['supernodal'] = 2

    # Obtain decomposition of M
    F = cholmod.symbolic(M)
    cholmod.numeric(M, F)

    # Restore previous options
    cholmod.options['supernodal'] = prevopts

    # As PMP'=LL' and det(P)=1 for all permutations
    # it follows det(M)=det(L)^2=product(diag(L))^2
    return(np.exp(2*sum(cvxopt.log(cholmod.diag(F)))))
    

    

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
# - perm: Input permutation (optional, one will be calculated if not)
# - retF: Return the factorisation object or not
# - retP: Return the permutation or not
# - retL: Return the lower cholesky or not
#
def sparse_chol(M, perm=None, retF=False, retP=True, retL=True):

    # Quick check that M is square
    if M.size[0]!=M.size[1]:
        raise Exception('M must be square.')

    # Set the factorisation to use LL' instead of LDL'
    cholmod.options['supernodal']=2

    if not perm is None:
        # Make an expression for the factorisation
        F=cholmod.symbolic(M,p=perm)
    else:
        # Make an expression for the factorisation
        F=cholmod.symbolic(M)

    # Calculate the factorisation
    cholmod.numeric(M, F)

    # Empty factorisation object
    factorisation = {}

    if retF:

        # Calculate the factorisation again (buggy if returning L for
        # some reason)
        F2=cholmod.symbolic(M,p=perm)
        cholmod.numeric(M, F2)

        # If we want to return the F object, add it to the dictionary
        factorisation['F']=F2

    if retP:

        # Set p to [0,...,n-1]
        P = cvxopt.matrix(range(M.size[0]), (M.size[0],1), tc='d')

        # Solve and replace p with the true permutation used
        cholmod.solve(F, P, sys=7)

        # Convert p into an integer array; more useful that way
        P=cvxopt.matrix(np.array(P).astype(np.int64),tc='i')

        # If we want to return the permutation, add it to the dictionary
        factorisation['P']=P

    if retL:

        # Get the sparse cholesky factor
        L=cholmod.getfactor(F)
        
        # If we want to return the factor, add it to the dictionary
        factorisation['L']=L

    # Return P and L
    return(factorisation)
    
# Z matrix generation function
# ----------------------------------------------
# This function takes in a dense matrix of
# parameters, a list of factors and a matrix of
# levels for said factors and generates the
# sparse rfx matrix Z.
# ----------------------------------------------
# The following inputs are required for this
# function:
#
# - params: These are the parameter columns that
#           will form the non-zero elements of Z.
# - factors: These are the grouping factors that
#            will be used for creating Z.
# - levels: This is a matrix of levels for
#           creating Z.
# ----------------------------------------------
# Example:
#
#    In R language the following model:
#
#        y ~ ffx + (rfx1|factor2) + ...
#               (rfx2|factor1) + (rfx3|factor3)
#
#    Would correspond to the following input
#    parameters:
#
#    - params = [ rfx1 | rfx2 | rfx3 ]
#    - factors = [2 1 3]
#    - levels = [ factor1 | factor2 | factor3 ]
def generate_Z(params,factors,levels):

    # TODO
    return(-1)

# PLS function
# ----------------------------------------------
# This function performs PLS to obtain the
# log likelihood value for parameter vector
# theta.
# ----------------------------------------------
# The following inputs are required for this
# function:
#
# - theta: The parameter estimate.
# - X: The fixed effects design matrix.
# - Z: The random effects design matrix.
# - Lambda: The lower cholesky factor of the
#           variance.
# - P: The sparse permutation for
#      Lamda'Z'ZLambda+I
# - nlevels: a vector of the number of levels
#            for each rfx grouping factor. e.g.
#            nlevels=[10,2] means there are 
#            10 levels for factor 1 and 2
#            levels for factor 2.
# - nparams: a vector of the number of rfx
#            variables for each grouping factor.
#            e.g. nlevels=[3,4] means there  
#            are 3 variables for factor 1 and 4
#            variables for factor 2.
def PLS(theta, X, Y, Z, P, nlevels, nparams):

    # Obtain Lambda from theta
    Lambda = mapping(theta, nlevels, nparams)

    # Obtain Lambda'Z'Y and Lambda'Z'X
    LambdatZt = spmatrix.trans(Lambda)*spmatrix.trans(Z)
    LambdatZtY = LambdatZt*Y
    LambdatZtX = LambdatZt*X

    # Set the factorisation to use LL' instead of LDL'
    cholmod.options['supernodal']=2

    # Obtain L
    LambdatZtZLambda = LambdatZt*spmatrix.trans(LambdatZt)
    I = spmatrix(1.0, range(Lambda.size[0]), range(Lambda.size[0]))
    chol_dict = sparse_chol(LambdatZtZLambda+I, perm=P, retF=True)
    L = chol_dict['L']
    F = chol_dict['F']

    # Obtain C_u (annoyingly solve writes over the second argument,
    # whereas spsolve outputs)
    Cu = LambdatZtY[P,:]
    cholmod.solve(F,Cu,sys=4)

    # Obtain RZX
    RZX = LambdatZtX[P,:]
    cholmod.solve(F,RZX,sys=4)

    # Obtain RXtRX
    RXtRX = matrix.trans(X)*X - matrix.trans(RZX)*RZX

    print(RXtRX.size)
    print(X.size)
    print(Y.size)
    print(RZX.size)
    print(Cu.size)
    

    # Obtain beta estimates (note: gesv also replaces the second
    # argument)
    betahat = matrix.trans(X)*Y - matrix.trans(RZX)*Cu
    lapack.gesv(RXtRX, betahat)

    # Obtain u estimates
    uhat = Cu-RZX*betahat
    cholmod.solve(F,uhat,sys=5)
    cholmod.solve(F,uhat,sys=8)

    # Obtain b estimates
    bhat = Lambda*uhat

    # Obtain y estimates
    Yhat = X*betahat + Z*bhat

    # Obtain residuals
    res = Y - Yhat

    # Obtain penalised residual sum of squares
    pss = matrix.trans(res)*res + matrix.trans(uhat)*uhat

    # Obtain Log(|L|^2)
    logdet = 2*sum(cvxopt.log(cholmod.diag(F))) # Need to do tr(R_X)^2 for rml



    return(betahat, bhat, Yhat, res, detlog)
    

# Examples
nparams = np.array([9,6,12,3,2,1])
nlevels = np.array([10,3,9,3,2,6])
theta = np.random.randn((np.sum(np.multiply(nparams,(nparams+1))/2)).astype(np.int64))
l=mapping(theta, nlevels, nparams)
print(inv_mapping(l)==theta)

# Go to test data directory
os.chdir('/home/tommaullin/Documents/BLMM-sandbox/testdata')

# Read in random effects variances
Z_3col=pd.read_csv('Z_3col.csv',header=None).values
Z = cvxopt.spmatrix(Z_3col[:,2].tolist(), (Z_3col[:,0]-1).astype(np.int64), (Z_3col[:,1]-1).astype(np.int64))
ZtZ=cvxopt.spmatrix.trans(Z)*Z
f = sparse_chol(ZtZ)
LLt=f['L']*cvxopt.spmatrix.trans(f['L'])
print(LLt-ZtZ[f['P'],f['P']])
print(sum(LLt-ZtZ[f['P'],f['P']]))

t1 = time.time()
sparse_chol(ZtZ)
t2 = time.time()
print(t2-t1)
t1 = time.time()
sparse_chol(ZtZ,perm=f['P'])
t2 = time.time()
print(t2-t1)

# Calculate lambda for R example
tmp =pd.read_csv('estd_rfxvar.csv',header=None).values
rfxvarest = spmatrix(tmp[tmp!=0],[0,0,1,1,2,2,3,3],[0,1,0,1,2,3,2,3])
f = sparse_chol(rfxvarest)
theta = inv_mapping(f['L'])

nlevels = np.array([20,3])
nparams = np.array([2,2])
Lam=mapping(theta, nlevels, nparams)
cvxopt.printing.options['width'] = -1

# Obtaining permutation for PLS
# Obtain Lambda'Z'ZLambda
LamtZt = spmatrix.trans(Lam)*spmatrix.trans(Z)
LamtZtZLam = LamtZt*spmatrix.trans(LamtZt)
f=sparse_chol(LamtZtZLam)
P = f['P']

Y=matrix(pd.read_csv('Y.csv',header=None).values)
X=matrix(pd.read_csv('X.csv',header=None).values)
t1 = time.time()
betahat, bhat, yhat, res,detlog =PLS(theta,X,Y,Z,P,nlevels,nparams)
t2 = time.time()

print(t1-t2)

# Determinant check
V = [10, 3, 5, -2, 5, 8,3,-2,3,3]
I = [0, 2, 1, 3, 2, 3,3,1,0,2]
J = [0, 0, 1, 1, 2, 3,2,3,2,3]
A = spmatrix(V,I,J)
A2 = np.zeros([4,4])
A2[I,J]=V
print(np.linalg.det(A2))
print(sp_det_sym(A))
print(detlog)
