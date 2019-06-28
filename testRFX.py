import numpy as np
from scipy import sparse

n_factors = 2; #Number of grouping factors, e.g. subject, site etc

f1_nl = 20; #Number of levels in factor 1, e.g. 20 for 20 subjects
f2_nl = 3; #Number of levels in factor 2, e.g. 3 for 3 sites

f1_nv = 2; #Number of variables for factor 1, e.g. 2 for subjects
f2_nv = 2; #Number of variables for factor 2, e.g. 2 for sites

N = 10000; # Number of Nifti volumes/dependent variables

# Construct FFX design matrix
x1 = np.ones([N,1]); # intercept
x2 = np.random.randn(N,1); # random covariate
x3 = np.random.randn(N,1); # random covariate
X = np.concatenate((x1,x2,x3),axis=1); # Full FFX matrix

# Construct RFX matrix for factor 1 (e.g. subject grouping)
z1_f1 = np.ones([N,1]); # intercept
z2_f1 = np.random.randn(N,1); # random covariate

# Construct RFX matrix for factor 2 (e.g. scan location grouping)
z1_f2 = np.ones([N,1]); # intercept
z2_f2 = np.random.randn(N,1); # random covariate

# Factoring
# e.g. 20 subjects each had 500 reading so first 500 for sub 1,
# 2nd 500 for sub 2 and so on.
f1 = np.kron(np.arange(1,(int(f1_nl)+1)),np.ones(int(N/f1_nl)))
# e.g. random factor for site location
f2 = np.random.choice(3,N)+1

# Sparse version of Z accounting for factoring factor 1:
row = np.kron(np.arange(0,N), np.ones(f1_nv)) # Row coordinates

# Column coordinates
f1tmp=f1.reshape(f1.shape[0],1)-1
coltmp = np.concatenate((f1_nv*f1tmp,f1_nv*f1tmp+1),axis=1)
col = coltmp.reshape(coltmp.shape[0]*coltmp.shape[1])

data = np.concatenate((z1_f1,z2_f1),axis=1).reshape(f1_nv*N)
Z1 = sparse.csr_matrix((data, (row, col)), shape=(N, f1_nl*f1_nv))
#np.savetxt('tmp.txt', Z1.toarray(), fmt='%.1f')

# Sparse version of Z accounting for factoring factor 2:
row = np.kron(np.arange(0,N), np.ones(f2_nv)) # Row coordinates

# Column coordinates
f2tmp=f2.reshape(f2.shape[0],1)-1
coltmp = np.concatenate((f2_nv*f2tmp,f2_nv*f2tmp+1),axis=1)
col = coltmp.reshape(coltmp.shape[0]*coltmp.shape[1])

data = np.concatenate((z1_f2,z2_f2),axis=1).reshape(f2_nv*N)
Z2 = sparse.csr_matrix((data, (row, col)), shape=(N, f2_nl*f2_nv))
#np.savetxt('tmp.txt', Z2.toarray(), fmt='%.1f')

Z = sparse.hstack([Z1,Z2])

# Generate positive semidefinite matrix for factor 1 covariance
random = np.random.rand(f1_nv, f1_nv)
sigma1 = np.dot(random, random.transpose())

# Generate positive semidefinite matrix for factor 2 covariance
random = np.random.rand(f2_nv, f2_nv)
sigma2 = np.dot(random, random.transpose())

# Make the sparse matrices
sigma1_full=sparse.kron(sparse.identity(f1_nl), sigma1)
sigma2_full=sparse.kron(sparse.identity(f2_nl), sigma2)

# Full sigma matrix
sigma=sparse.block_diag((sigma1_full,sigma2_full))

# Generate response 
