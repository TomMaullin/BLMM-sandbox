from cvxopt import matrix, spmatrix, cholmod

A = spmatrix([10, 3, 5, -2, 5, 2], [0, 2, 1, 3, 2, 3], [0, 0, 1, 1, 2, 3])
X = matrix(range(8), (4,2), 'd')
F = cholmod.symbolic(A)
cholmod.numeric(A, F)
cholmod.solve(F, X)
print(X)
P = cvxopt.matrix(P)
print(cvxopt.matrix(list(map(lambda x: x != 1, abs(P))), P.size))
