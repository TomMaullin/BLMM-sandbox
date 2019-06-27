library(MASS)
library(Matrix)
library(lme4)

# Say 20 subs, 5 readings each
fS <- gl(20, 50)
JS <- t(as(fS, Class = "sparseMatrix"))
num_s <- 5

# Random intercept and random readings as RFX
XS <- cbind(1,20*rnorm(1000))

# Generate random effects matrix for subject factor
ZS <- t(KhatriRao(t(JS), t(XS)))

# Construct RFX matrix
Z <- ZS

# Image of Z'Z
image(t(Z)%*%Z)

# Image of Zi'Zi where Z1 is the first 10 rows of Z, 
# z2 is second and so on
#image(t(Z[1:10,])%*%Z[1:10,])
#image(t(Z[11:20,])%*%Z[11:20,])

# Fixed effects matrix
X <- cbind(1, rnorm(1000), rnorm(1000))

# Make RFX variance matrix

# function for positive definite matrix
# For subject rfx
p <- qr.Q(qr(matrix(rnorm(2^2), 2)))
cov_s <- crossprod(p, p*(2:1))

# For group rfx
p <- qr.Q(qr(matrix(rnorm(2^2), 2)))
cov_g <- crossprod(p, p*(2:1))

# Combine to get sparse block diag
sigma <- cov_s
for (i in c(2:20)){
  sigma <- bdiag(sigma, cov_s)
}

# Generate b
b <- mvrnorm(n = 1, matrix(0L, nrow = dim(Z)[2], ncol = 1), sigma, tol = 1e-6, empirical = FALSE, EISPACK = FALSE)

# Generate beta
beta <- c(1:3)

# Generate response 
y <- X%*%beta + Z%*%b + rnorm(1000)

#------------------------------------------------------------------------------------------------
# Now to see if lmer can decipher this model
y <- as.matrix(y)

x1 <- as.matrix(X[,1])
x2 <- as.matrix(X[,2])
x3 <- as.matrix(X[,3])

z1 <- as.matrix(XS[,1])
z2 <- as.matrix(XS[,2])

m <- lmer(y ~ x2 + x3 + (z2|fS)) #Don't need intercepts in R - automatically assumed

# FFX estimates
fixef(m)
print(beta)

# RFX estimates
ranef(m)
print(t(matrix(b, 2, 20)))

# RFX variances
as.matrix(Matrix::bdiag(VarCorr(m)))
print(cov_s)
print(cov_g)

summary(m)

