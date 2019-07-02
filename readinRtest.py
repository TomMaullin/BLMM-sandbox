import numpy as np
from scipy import sparse
import pandas
import os
from scipy import linalg
import cvxopt
from cvxopt import cholmod, umfpack, amd
import time
import pandas as pd

cvxopt.printing.options['width'] = -3

# In this example the number of dependent variables is 1000
N=1000

# There are 2 grouping factors
nf=2

# One grouping factor has 20 levels and one with 3 levels
f1_nl=20
f2_nl=3

# Both grouping factors have 2 dependent varibles
f1_nv=2
f2_nv=2

# The q dimension is given by
q=f1_nl*f1_nv+f2_nl*f2_nv

# Go to test data directory
os.chdir('/home/tommaullin/Documents/BLMM-sandbox/testdata')

# Read in beta
beta=pandas.read_csv('true_beta.csv',header=None).values

# Read in random effects variances
D_3col=pandas.read_csv('true_rfxvar_3col.csv',header=None).values
D = cvxopt.spmatrix(D_3col[:,2].tolist(), (D_3col[:,0]-1).astype(np.int64), (D_3col[:,1]-1).astype(np.int64))

# Read in Z
Z_3col=pandas.read_csv('Z_3col.csv',header=None).values
t1 = time.time()
Z = sparse.csr_matrix((Z_3col[:,2], (Z_3col[:,0]-1, Z_3col[:,1]-1)), shape=(N, q))
t2 = time.time()
print(t1-t2)

t1 = time.time()
Z2 = cvxopt.spmatrix(Z_3col[:,2].tolist(), (Z_3col[:,0]-1).astype(np.int64), (Z_3col[:,1]-1).astype(np.int64))
t2 = time.time()
print(t1-t2)

# Read in X
X=pandas.read_csv('X.csv',header=None).values

# Read in b
b=pandas.read_csv('true_b.csv',header=None).values


print(D)
# Read in Y
Y=pandas.read_csv('Y.csv',header=None).values

# Read in ffx variance
sigma2=pandas.read_csv('true_ffxvar.csv',header=None).values

# Use minimum degree ordering
P=amd.order(D)

# Set the factorisation to use LL' instead of LDL'
cholmod.options['supernodal']=2

# Make an expression for the factorisation
F=cholmod.symbolic(D,p=P)

# Calculate the factorisation
cholmod.numeric(D, F) 

# Get the sparse cholesky factorisation
L=cholmod.getfactor(F)
LLt=L*cvxopt.spmatrix.trans(L)

# Create initial lambda
# Work out how many lambda components needed
f1_n_lamcomps = np.int64(f1_nv*(f1_nv+1)/2)
f2_n_lamcomps = np.int64(f2_nv*(f2_nv+1)/2)

# Work out triangular indices for lambda (blocks starting from (0,0))
r_inds_f1_block, c_inds_f1_block = np.tril_indices(f1_nv)
r_inds_f2_block, c_inds_f2_block = np.tril_indices(f2_nv)

# Repeat the factor 1 block for each level of factor 1
r_inds_f1 = r_inds_f1_block
c_inds_f1 = c_inds_f1_block
for i in range(1,f1_nl):
    r_inds_f1 = np.hstack((r_inds_f1,r_inds_f1_block+(i*f1_nv)))
    c_inds_f1 = np.hstack((c_inds_f1,c_inds_f1_block+(i*f1_nv)))

# Repeat the factor 2 block for each level of factor 2
r_inds_f2 = r_inds_f2_block+(f1_nl*f1_nv)
c_inds_f2 = c_inds_f2_block+(f1_nl*f1_nv)
for i in range(1,f2_nl):
    r_inds_f2 = np.hstack((r_inds_f2,r_inds_f2_block+(i*f2_nv)+(f1_nl*f1_nv)))
    c_inds_f2 = np.hstack((c_inds_f2,c_inds_f2_block+(i*f2_nv)+(f1_nl*f1_nv)))

# print(cvxopt.spmatrix(np.ones(60).tolist(), r_inds.astype(np.int64), c_inds.astype(np.int64)))

theta = np.random.randn(np.int64(f1_n_lamcomps+f2_n_lamcomps))

theta_repeated = np.hstack((np.tile(theta[0:f1_n_lamcomps], f1_nl),np.tile(theta[f1_n_lamcomps:(f1_n_lamcomps+f2_n_lamcomps)], f2_nl)))

lambda_theta = cvxopt.spmatrix(theta_repeated.tolist(), np.hstack((r_inds_f1,r_inds_f2)).astype(np.int64), np.hstack((c_inds_f1,c_inds_f2)).astype(np.int64))

theta_fromlambda = pd.unique(list(lambda_theta))
