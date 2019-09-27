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
cov_s <- matrix(c(2,0.5,0.5,4),nrow=2,ncol=2)

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

m <- lmer(y ~ x2 + x3 + (z2|fS), REML=FALSE) #Don't need intercepts in R - automatically assumed

# FFX estimates
fixef(m)
print(beta)

# RFX estimates
ranef(m)
print(t(matrix(b, 2, 20)))

# RFX variances
as.matrix(Matrix::bdiag(VarCorr(m)))
print(cov_s)

summary(m)

setwd('C:/Users/TomM/Documents/BLMM-testdata')

# 3 column format for Z
Z_3col<-as.data.frame(summary(Z))
colnames(Z_3col) <- NULL
write.csv(Z_3col,file="./Z_3col_1factor.csv",row.names=FALSE)

sigma_3col<-as.data.frame(summary(sigma))
colnames(sigma_3col) <- NULL
write.csv(sigma_3col,file="./true_rfxvar_3col_1factor.csv",row.names=FALSE)

X <- as.data.frame(X)
colnames(X)<-NULL
write.csv(X,file="./X_1factor.csv",row.names=FALSE)

y <- as.data.frame(y)
colnames(y)<-NULL
write.csv(y,file="./Y_1factor.csv",row.names=FALSE)

beta <- as.data.frame(beta)
colnames(beta)<-NULL
write.csv(beta,file="./true_beta_1factor.csv",row.names=FALSE)


true_rfx <- data.frame(b)
colnames(true_rfx) <- NULL
write.csv(true_rfx,file="./true_b_1factor.csv",row.names=FALSE)

true_ffxvar <- data.frame(c(1))
colnames(true_ffxvar) <- NULL
write.csv(true_ffxvar,file="./true_ffxvar_1factor.csv",row.names=FALSE)

# Save the estimates as well
rfx_1<-ranef(m)$fS
colnames(rfx_1) <- ''
est_rfx <- rfx_1
colnames(est_rfx) <- NULL
write.csv(est_rfx,file="./estd_b_1factor.csv",row.names=FALSE)

rfxvar_est <- data.frame(as.matrix(Matrix::bdiag(VarCorr(m))))
colnames(rfxvar_est) <- NULL
write.csv(rfxvar_est,file="./estd_rfxvar_1factor.csv",row.names=FALSE)

est_ffx <- data.frame(fixef(m))
colnames(est_ffx) <- NULL
write.csv(est_ffx,file="./estd_beta_1factor.csv",row.names=FALSE)

est_ll <- data.frame(logLik(m))
colnames(est_ll) <- NULL
write.csv(est_ll,file="./estd_ll_1factor.csv",row.names=FALSE)


