import numpy as np
from scipy import sparse
import pandas
import os
from scipy import linalg
import cvxopt
from cvxopt import cholmod
import time

cvxopt.printing.options['width'] = -3

# In this example the number of dependent variables is 1000
N=1000

# There are 2 grouping factors, one with 20 levels and one with 3 levels
f1_nl=20
f2_nl=3

# Both grouping factors have 2 dependent varibles
f1_nv=2
f2_nv=2

# The q dimension is given by
q=f1_nl*f1_nv+f2_nl*f2_nv

# Go to test data directory
os.chdir('/home/tommaullin/BLMM-sandbox/testdata')

# Read in beta
beta=pandas.read_csv('true_beta.csv',header=None).values

# Read in random effects variances
D_3col=pandas.read_csv('true_rfxvar_3col.csv',header=None).values
D = sparse.csr_matrix((D_3col[:,2], (D_3col[:,0]-1, D_3col[:,1]-1)), shape=(q, q))

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


ZtZ=cvxopt.spmatrix.trans(Z2)*Z2
print(ZtZ)
# Read in Y
Y=pandas.read_csv('Y.csv',header=None).values

# Read in ffx variance
sigma2=pandas.read_csv('true_ffxvar.csv',header=None).values

cholmod.options['supernodal']=2
F=cholmod.symbolic(ZtZ) # Makes expression for factorisation
cholmod.numeric(ZtZ, F) # Numerically calculates and puts result in F
ZtZ2=cvxopt.matrix(ZtZ)

cholmod.solve(F, ZtZ2, sys=2)

