import numpy as np
from scipy import sparse
import pandas
import os

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
Z = sparse.csr_matrix((Z_3col[:,2], (Z_3col[:,0]-1, Z_3col[:,1]-1)), shape=(N, q))

# Read in X
X=pandas.read_csv('X.csv',header=None).values

# Read in b
b=pandas.read_csv('true_b.csv',header=None).values

# Read in Y
Y=pandas.read_csv('Y.csv',header=None).values

# Read in ffx variance
sigma2=pandas.read_csv('true_ffxvar.csv',header=None).values
