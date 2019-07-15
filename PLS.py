import numpy as np
import cvxopt
from cvxopt import cholmod, umfpack, amd, matrix, spmatrix, lapack
import pandas as pd
from scipy.optimize import minimize
###
import os
import time
import numba
import sparse #All functions but the chol should be converted to use this

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
def get_mapping(theta, nlevels, nparams):

    # Work out how many factors there are
    n_f = len(nlevels)

    # Quick check that nlevels and nparams are the same length
    #if len(nlevels)!=len(nparams):
    #    raise Exception('The number of parameters and number of levels should be recorded for every grouping factor.')

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
    theta_repeated_inds = np.array([])
    
    # Loop through factors generating the indices to map theta to.
    for i in range(0,n_f):

        # Work out the indices of a lower triangular matrix
        # of size #variables(factor) by #variables(factor)
        row_inds_tri, col_inds_tri = np.tril_indices(nparams[i])

        # Work out theta for this block
        theta_current_inds = np.arange(np.sum(n_lamcomps[0:i]),np.sum(n_lamcomps[0:(i+1)]))

        # Work out the repeated theta
        theta_repeated_inds = np.hstack((theta_repeated_inds, np.tile(theta_current_inds, nlevels[i])))

        # For each level of the factor we must repeat the lower
        # triangular matrix
        for j in range(0,nlevels[i]):

            # Append the row/column indices to the running list
            row_indices = np.hstack((row_indices, (row_inds_tri+block_index)))
            col_indices = np.hstack((col_indices, (col_inds_tri+block_index)))

            # Move onto the next block
            block_index = block_index + nparams[i]

    # Create lambda as a sparse matrix
    #lambda_theta = spmatrix(theta_repeated.tolist(), row_indices.astype(np.int64), col_indices.astype(np.int64))

    # Return lambda
    return(theta_repeated_inds, row_indices, col_indices)

def mapping(theta, theta_inds, r_inds, c_inds):

    return(spmatrix(theta[theta_inds.astype(np.int64)].tolist(), r_inds.astype(np.int64), c_inds.astype(np.int64)))
    

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
def PLS(theta, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, P, tinds, rinds, cinds):

    #t1 = time.time()
    # Obtain Lambda from theta
    Lambda = mapping(theta, tinds, rinds, cinds)
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    # Obtain Lambda'
    Lambdat = spmatrix.trans(Lambda)
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    LambdatZtY = Lambdat*ZtY
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    LambdatZtX = Lambdat*ZtX
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    # Set the factorisation to use LL' instead of LDL'
    cholmod.options['supernodal']=2
    #t2 = time.time()
    #print(t2-t1)

    # Obtain L
    #t1 = time.time()
    LambdatZtZLambda = Lambdat*ZtZ*Lambda
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    I = spmatrix(1.0, range(Lambda.size[0]), range(Lambda.size[0]))
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    chol_dict = sparse_chol(LambdatZtZLambda+I, perm=P, retF=True)
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    L = chol_dict['L']
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    F = chol_dict['F']
    #t2 = time.time()
    #print(t2-t1)

    # Obtain C_u (annoyingly solve writes over the second argument,
    # whereas spsolve outputs)
    #t1 = time.time()
    Cu = LambdatZtY[P,:]
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    cholmod.solve(F,Cu,sys=4)
    #t2 = time.time()
    #print(t2-t1)

    # Obtain RZX
    #t1 = time.time()
    RZX = LambdatZtX[P,:]
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    cholmod.solve(F,RZX,sys=4)
    #t2 = time.time()
    #print(t2-t1)

    # Obtain RXtRX
    #t1 = time.time()
    RXtRX = XtX - matrix.trans(RZX)*RZX
    #t2 = time.time()
    #print(t2-t1)

    #print(RXtRX.size)
    #print(X.size)
    #print(Y.size)
    #print(RZX.size)
    #print(Cu.size)
    

    # Obtain beta estimates (note: gesv also replaces the second
    # argument)
    #t1 = time.time()
    betahat = XtY - matrix.trans(RZX)*Cu
    #t2 = time.time()
    #print(t2-t1)
    
    #t1 = time.time()
    lapack.gesv(RXtRX, betahat)
    #t2 = time.time()
    #print(t2-t1)

    # Obtain u estimates
    #t1 = time.time()
    uhat = Cu-RZX*betahat
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    cholmod.solve(F,uhat,sys=5)
    #t2 = time.time()
    #print(t2-t1)

    #t1 = time.time()
    cholmod.solve(F,uhat,sys=8)
    #t2 = time.time()
    #print(t2-t1)

    # Obtain b estimates
    #t1 = time.time()
    bhat = Lambda*uhat
    #t2 = time.time()
    #print(t2-t1)

    # Obtain residuals sum of squares
    #t1 = time.time()
    resss = YtY-2*YtX*betahat-2*YtZ*bhat+2*matrix.trans(betahat)*XtZ*bhat+matrix.trans(betahat)*XtX*betahat+matrix.trans(bhat)*ZtZ*bhat
    #t2 = time.time()
    #print(t2-t1)

    # Obtain penalised residual sum of squares
    #t1 = time.time()
    pss = resss + matrix.trans(uhat)*uhat
    #t2 = time.time()
    #print(t2-t1)

    # Obtain Log(|L|^2)
    #t1 = time.time()
    logdet = 2*sum(cvxopt.log(cholmod.diag(F))) # this method only works for symm decomps
    # Need to do tr(R_X)^2 for rml
    #t2 = time.time()
    #print(t2-t1)

    # Obtain log likelihood
    logllh = -logdet/2-X.size[0]/2*(1+np.log(2*np.pi*pss)-np.log(X.size[0]))

    #print(L[::(L.size[0]+1)]) # gives diag
    #print(logllh[0,0])
    #print(theta)

    return(-logllh[0,0])
    

# Examples
nparams = np.array([9,6,12,3,2,1])
nlevels = np.array([10,3,9,3,2,6])
theta = np.random.randn((np.sum(np.multiply(nparams,(nparams+1))/2)).astype(np.int64))
#l=get_mapping(theta, nlevels, nparams)
#print(inv_mapping(l)==theta)

# Go to test data directory
os.chdir('/home/tommaullin/Documents/BLMM-sandbox/testdata')

# Read in random effects variances
Z_3col=pd.read_csv('Z_3col.csv',header=None).values
Z = cvxopt.spmatrix(Z_3col[:,2].tolist(), (Z_3col[:,0]-1).astype(np.int64), (Z_3col[:,1]-1).astype(np.int64))
ZtZ=cvxopt.spmatrix.trans(Z)*Z
f = sparse_chol(ZtZ)
LLt=f['L']*cvxopt.spmatrix.trans(f['L'])
#print(LLt-ZtZ[f['P'],f['P']])
#print(sum(LLt-ZtZ[f['P'],f['P']]))

t1 = time.time()
sparse_chol(ZtZ)
t2 = time.time()
#print(t2-t1)
t1 = time.time()
sparse_chol(ZtZ,perm=f['P'])
t2 = time.time()
#print(t2-t1)

# Calculate lambda for R example
tmp =pd.read_csv('estd_rfxvar.csv',header=None).values
rfxvarest = spmatrix(tmp[tmp!=0],[0,0,1,1,2,2,3,3],[0,1,0,1,2,3,2,3])
f = sparse_chol(rfxvarest)
theta = inv_mapping(f['L'])

nlevels = np.array([20,3])
nparams = np.array([2,2])
tinds,rinds,cinds=get_mapping(theta, nlevels, nparams)
Lam=mapping(theta,tinds,rinds,cinds)
#cvxopt.printing.options['width'] = -1

# Obtaining permutation for PLS
# Obtain Lambda'Z'ZLambda
LamtZt = spmatrix.trans(Lam)*spmatrix.trans(Z)
LamtZtZLam = LamtZt*spmatrix.trans(LamtZt)
#f=sparse_chol(LamtZtZLam)
#P = f['P']
P=cvxopt.amd.order(LamtZtZLam)

Y=matrix(pd.read_csv('Y.csv',header=None).values)
X=matrix(pd.read_csv('X.csv',header=None).values)

ZtX=cvxopt.spmatrix.trans(Z)*X
ZtY=cvxopt.spmatrix.trans(Z)*Y
XtX=cvxopt.matrix.trans(X)*X
ZtZ=cvxopt.spmatrix.trans(Z)*Z
XtY=cvxopt.matrix.trans(X)*Y
YtX=cvxopt.matrix.trans(Y)*X
YtZ=cvxopt.matrix.trans(Y)*Z
XtZ=cvxopt.matrix.trans(X)*Z
YtY=cvxopt.matrix.trans(Y)*Y

tinds, rinds, cinds = get_mapping(theta, nlevels, nparams)

t1 = time.time()
estllh =-PLS(theta,ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY,P,tinds, rinds, cinds)
t2 = time.time()

truellh=matrix(pd.read_csv('./estd_ll.csv',header=None).values)[0]

#print(t2-t1)

# Determinant check
V = [10, 3, 5, -2, 5, 8,3,-2,3,3]
I = [0, 2, 1, 3, 2, 3,3,1,0,2]
J = [0, 0, 1, 1, 2, 3,2,3,2,3]
A = spmatrix(V,I,J)
A2 = np.zeros([4,4])
A2[I,J]=V
#print(np.linalg.det(A2))
#print(sp_det_sym(A))
#print(logdet)

theta0=np.array([1,0,1,1,0,1])

t1 = time.time()
#theta_est=minimize(PLS, theta0, args=(ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY ,P,nlevels,nparams), method='Nelder-Mead', tol=1e-6)
t2 = time.time()

#print(t2-t1)


t1 = time.time()
theta_est=minimize(PLS, theta0, args=(ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY ,P,tinds, rinds, cinds), method='Powell', tol=1e-6)
t2 = time.time()

print(t2-t1)
