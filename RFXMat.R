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

# Say 3 sites/groups, subjects randomly scanned at
fG <- as.factor(sample(c(1:3),1000,replace=TRUE))
JG <- t(as(fG, Class = "sparseMatrix"))

# Random intercept and random readings as RFX
XG <- cbind(1,10*rnorm(1000))
num_g <- 2

# Generate random effects matrix for group factor
ZG <- t(KhatriRao(t(JG), t(XG)))

# Construct RFX matrix
Z <- cbind(ZS, ZG)

# Image of Z'Z
image(t(Z)%*%Z)

# Image of Zi'Zi where Z1 is the first 10 rows of Z, 
# z2 is second and so on
#image(t(Z[1:10,])%*%Z[1:10,])
#image(t(Z[11:20,])%*%Z[11:20,])

# Fixed effects matrix
X <- cbind(1, rnorm(1000), rnorm(1000))

# Make RFX variance matrix
cov_s <- matrix(c(2,0.5,0.5,4),nrow=2,ncol=2)
cov_g <- matrix(c(6,0.1,0.1,1),nrow=2,ncol=2)

# Combine to get sparse block diag
sigma <- cov_s
for (i in c(2:20)){
  sigma <- bdiag(sigma, cov_s)
}
for (i in c(1:3)){
  sigma <- bdiag(sigma, cov_g)
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
z3 <- as.matrix(XG[,1])
z4 <- as.matrix(XG[,2])

m <- lmer(y ~ x2 + x3 + (z2|fS) + (z4|fG)) #Don't need intercepts in R - automatically assumed

# FFX estimates
fixef(m)
print(beta)

# RFX estimates
ranef(m)
print(t(matrix(b, 2, 23)))

# RFX variances 
as.matrix(Matrix::bdiag(VarCorr(m)))
print(cov_s)
print(cov_g)

summary(m)

# 3 column format for Z
Z_3col<-as.data.frame(summary(Z))
colnames(Z_3col) <- NULL
write.csv(Z_3col,file="./testdata/Z_3col.csv",row.names=FALSE)

sigma_3col<-as.data.frame(summary(sigma))
colnames(sigma_3col) <- NULL
write.csv(sigma_3col,file="./testdata/true_rfxvar_3col.csv",row.names=FALSE)

X <- as.data.frame(X)
colnames(X)<-NULL
write.csv(X,file="./testdata/X.csv",row.names=FALSE)

y <- as.data.frame(y)
colnames(y)<-NULL
write.csv(y,file="./testdata/Y.csv",row.names=FALSE)

beta <- as.data.frame(beta)
colnames(beta)<-NULL
write.csv(beta,file="./testdata/true_beta.csv",row.names=FALSE)


true_rfx <- data.frame(b)
colnames(true_rfx) <- NULL
write.csv(true_rfx,file="./testdata/true_b.csv",row.names=FALSE)

write.csv(c(1),file="./testdata/true_ffxvar.csv",row.names=FALSE)

# Save the estimates as well
rfx_1<-ranef(m)$fS
rfx_2<-ranef(m)$fG
colnames(rfx_1) <- ''
colnames(rfx_2) <- ''
est_rfx <- rbind(rfx_1,rfx_2)
colnames(est_rfx) <- NULL
write.csv(est_rfx,file="./testdata/estd_b.csv",row.names=FALSE)

rfxvar_est <- data.frame(as.matrix(Matrix::bdiag(VarCorr(m))))
colnames(rfxvar_est) <- NULL
write.csv(rfxvar_est,file="./testdata/estd_rfxvar.csv",row.names=FALSE)

est_ffx <- data.frame(fixef(m))
colnames(est_ffx) <- NULL
write.csv(est_ffx,file="./testdata/estd_beta.csv",row.names=FALSE)
