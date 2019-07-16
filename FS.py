import numpy as np
import cvxopt
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import sys
np.set_printoptions(threshold=sys.maxsize)
###
import os
import time
import numba
import sparse #All functions but the chol should be converted to use this

def FS(beta0, sigma0, D0, ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY, P, tinds, rinds, cinds):

    return(1)


N=1000

nlevels = np.array([20,3])
nparams = np.array([2,2])

# Go to test data directory
os.chdir('/home/tommaullin/BLMM-sandbox/testdata')

# Read in random effects variances
Z_3col=pd.read_csv('Z_3col.csv',header=None).values
Z = csr_matrix((Z_3col[:,2].tolist(), ((Z_3col[:,0]-1).astype(np.int64), (Z_3col[:,1]-1).astype(np.int64))))
ZtZ=Z.transpose()*Z

Z = csr_matrix((Z_3col[:,2].tolist(), ((Z_3col[:,0]-1).astype(np.int64), (Z_3col[:,1]-1).astype(np.int64))))

Z2 = sparse.COO([(Z_3col[:,0]-1).astype(np.int64), (Z_3col[:,1]-1).astype(np.int64)], Z_3col[:,2].tolist(), shape=(N, 46))

# Calculate lambda for R example
tmp =pd.read_csv('estd_rfxvar.csv',header=None).values
#rfxvarest = spmatrix(tmp[tmp!=0],[0,0,1,1,2,2,3,3],[0,1,0,1,2,3,2,3])

#cvxopt.printing.options['width'] = -1

#Y=matrix(pd.read_csv('Y.csv',header=None).values)
#X=matrix(pd.read_csv('X.csv',header=None).values)

#ZtX=cvxopt.spmatrix.trans(Z)*X
#ZtY=cvxopt.spmatrix.trans(Z)*Y
#XtX=cvxopt.matrix.trans(X)*X
#ZtZ=cvxopt.spmatrix.trans(Z)*Z
#XtY=cvxopt.matrix.trans(X)*Y
#YtX=cvxopt.matrix.trans(Y)*X
#YtZ=cvxopt.matrix.trans(Y)*Z
#XtZ=cvxopt.matrix.trans(X)*Z
#YtY=cvxopt.matrix.trans(Y)*Y

#t1 = time.time()
#estllh =-PLS(theta,ZtX, ZtY, XtX, ZtZ, XtY, YtX, YtZ, XtZ, YtY,P,tinds, rinds, cinds)
#t2 = time.time()

#truellh=matrix(pd.read_csv('./estd_ll.csv',header=None).values)[0]


