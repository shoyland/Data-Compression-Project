#STAT 672 Final Project
#Data Compression
#Sparse Stochastic Gradient Descent

#logistic regression - batch gradient descent
#test dataset
library(ISLR)
names(Smarket)
data(Smarket)

#X is an nxd matrix
#y is an nx1 matrix
#n_iter is a constant number of interations
batch_grad_des <- function(Learning_Rate,n_iter,X,y){
  #Step 1: Initialize Parameter
  runtime <- proc.time()[3]
  N <- nrow(X)
  w <- matrix(0,nrow=ncol(X),ncol=1)
  b <- 0
  cost <- 0
  #dim <- c(0,0)
  #starting loop
  for (i in 1:n_iter){
    #Step 2: Apply Sigmoid Function and get y prediction
    Z <- t(t(w)%*%t(X)+b)
    y_pred = 1/(1+1/exp(Z))
    
    #Step 3: Calculate Loss Function
    cost = -(1/N)*sum(y*log(y_pred)+(1-y)*log(1-y_pred))
    
    #Step 4: Calculate Gradient
    dw <- 1/N * t(X)%*%(y_pred-y)
    db <- 1/N * sum(y_pred-y)
    
    #Step 5: Update intercept (b) and coefficients (W)
    w <- w - Learning_Rate*dw
    b <- b - Learning_Rate*db
    
    #dim <- dim(as.matrix(cost))
  }
  #return coefficients
  #return(dim)
  runtime <- proc.time()[3]-runtime
  return(round(rbind(b,w,cost,runtime),3))
}
data <- Smarket
data <- data[,-1]
Xdata <- as.matrix(data[,-7:-8])
Ydata <- as.matrix(as.numeric(data[,8]))-1
batch_grad_des(Learning_Rate = 0.01,n_iter=10,X=Xdata,y=Ydata)

stoch_grad_des <- function(Learning_Rate,n_iter,X,y){
  #Step 1: Initialize Parameters
  runtime <- proc.time()[3]
  N <- nrow(X)
  w <- matrix(0,nrow=ncol(X),ncol=1)
  b <- 0
  cost <- 0
  for ( i in 1:n_iter){
    for (j in 1:N){
      #choose 1 record
      XX <- as.matrix(t(X[j,]))
      yy <- as.numeric(y[j,1])
      #Step 2: Apply Sigmoid Function and get y prediction
      Z <- t(w)%*%t(XX)+b
      y_pred = as.numeric(1/(1+1/exp(Z)))
      
      #Step 3: Calculate Loss Function
      cost = -(yy*log(y_pred)+(1-yy)*log(1-y_pred))
      
      #Step 4: Calculate Gradient
      dw <- t(XX) * y_pred
      db <- y_pred - yy
      
      #Step 5: Update intercept (b) and coefficients (W)
      w <- w - Learning_Rate*dw
      b <- b - Learning_Rate*db 
    }
    #Calculate loss function
    Z_full <- t(t(w)%*%t(X)+b)
    y_pred_full = 1/(1+1/exp(Z_full))
    total_cost = -(1/N)*sum(y*log(y_pred_full)+(1-y)*log(1-y_pred_full)) 
  }
  #return(dw)
  runtime <- proc.time()[3]-runtime
  return(round(rbind(b,w,total_cost,runtime),3))
}
stoch_grad_des(Learning_Rate = 0.01,n_iter=2000,X=Xdata,y=Ydata)

library(foreach)
library(doParallel)
dist_stoch_grad_des <- function(Learning_Rate,n_iter,X,y){
  #Step 1: Initialize Parameters
  runtime <- proc.time()[3]
  w_names <- as.matrix(X[1,])
  N <- nrow(X)
  w <- matrix(0,nrow=ncol(X),ncol=1)
  b <- 0
  cost <- 0
  average <- matrix(0,nrow=(nrow(w)+1),ncol=1)
  combin <- matrix(0,nrow=(nrow(w)+1),ncol=numCors)
  for ( k in 1: n_iter){
    combin <- foreach ( i = 1:numCors) %dopar% {
      #Step 6: Update intercept (b) and coefficients (W)
      w <- w - Learning_Rate*as.matrix(average[-nrow(average),1])
      b <- b - Learning_Rate*as.numeric(average[nrow(average),1])
      #Step 1a: subset X
      subset <- sample(1:nrow(X),size = round(0.33*nrow(X)), replace=F)
      X_sample <- X[subset,]
      y_sample <- y[subset,]
      
      #Step 2: Apply Sigmoid Function and get y prediction
      Z <- t(t(w)%*%t(X_sample)+b)
      y_pred = 1/(1+1/exp(Z))
      
      #Step 3: Calculate Loss Function
      cost = -(1/N)*sum(y_sample*log(y_pred)+(1-y_sample)*log(1-y_pred))
      
      #Step 4: Calculate Gradient
      dw <- 1/N * t(X_sample)%*%(y_pred-y_sample)
      db <- 1/N * sum(y_pred-y_sample)
      
      combin <- rbind(dw,db)
      combin
    }
    average <- matrix(rowMeans(matrix(unlist(combin,F),nrow=nrow(combin[[1]]),ncol=numCors)),ncol=1)
  }
  #Step 5: Update
  w <- as.matrix(average[-nrow(average),1])
  b <- as.numeric(average[nrow(average),1])
  #Calculate loss function
  Z_full <- t(t(w)%*%t(X)+b)
  y_pred_full = 1/(1+1/exp(Z_full))
  total_cost = -(1/N)*sum(y*log(y_pred_full)+(1-y)*log(1-y_pred_full)) 
  
  #return(dim(w_names))
  w_names[,1] <- w[,1]
  runtime <- proc.time()[3]-runtime
  return(round(rbind(b,w_names,total_cost,runtime),3))
  
}
numCors <- 3
registerDoParallel(numCors)
dist_stoch_grad_des(Learning_Rate = 0.01,n_iter=2000,X=Xdata,y=Ydata)
stopImplicitCluster()

#sparse stochastic gradient - distributed
library(foreach)
library(doParallel)
dist_sparse_stoch_grad_des <- function(Learning_Rate,n_iter,X,y){
  #Step 1: Initialize Parameters
  runtime <- proc.time()[3]
  w_names <- as.matrix(X[1,])
  N <- nrow(X)
  w <- matrix(0,nrow=ncol(X),ncol=1)
  p <- w
  Z <- p
  Q.w <- Z
  b <- 0
  cost <- 0
  average <- matrix(0,nrow=(nrow(w)+1),ncol=1)
  combin <- matrix(0,nrow=(nrow(w)+1),ncol=numCors)
  for ( k in 1: n_iter){
    combin <- foreach ( i = 1:numCors) %dopar% {
      #Step 6: Update intercept (b) and coefficients (W)
      w <- w - Learning_Rate*as.matrix(average[-nrow(average),1])
      b <- b - Learning_Rate*as.numeric(average[nrow(average),1])
      #Step 1a: subset X
      subset <- sample(1:nrow(X),size = round(0.33*nrow(X)), replace=F)
      X_sample <- X[subset,]
      y_sample <- y[subset,]
      
      #Step 2: Apply Sigmoid Function and get y prediction
      Z <- t(t(w)%*%t(X_sample)+b)
      y_pred = 1/(1+1/exp(Z))
      
      #Step 3: Calculate Loss Function
      cost = -(1/N)*sum(y_sample*log(y_pred)+(1-y_sample)*log(1-y_pred))
      
      #Step 4: Calculate Gradient
      dw <- 1/N * t(X_sample)%*%(y_pred-y_sample)
      db <- 1/N * sum(y_pred-y_sample)
      
      #Step 5: Sparsify
      p <- as.matrix(runif(nrow(p),0,1))
      for ( d in 1:nrow(p)){
        Z[d,1] <- rbinom(1,size=1,prob=p[d,1])
        Q.w[d,1]<-Z[d,1]*dw[d,1]/p[d,1]
      }
      
      combin <- rbind(Q.w,db)
      combin
    }
    average <- matrix(rowMeans(matrix(unlist(combin,F),nrow=nrow(combin[[1]]),ncol=numCors)),ncol=1)
  }
  w <- as.matrix(average[-nrow(average),1])
  b <- as.numeric(average[nrow(average),1])
  #Calculate loss function
  Z_full <- t(t(w)%*%t(X)+b)
  y_pred_full = 1/(1+1/exp(Z_full))
  total_cost = -(1/N)*sum(y*log(y_pred_full)+(1-y)*log(1-y_pred_full)) 
  
  #return(dim(w_names))
  w_names[,1] <- w[,1]
  runtime <- proc.time()[3]-runtime
  return(round(rbind(b,w_names,total_cost,runtime),3))
}

numCors <- 4
registerDoParallel(numCors)
dist_stoch_grad_des(Learning_Rate = 0.01,n_iter=3000,X=Xdata,y=Ydata)
dist_sparse_stoch_grad_des(Learning_Rate = 0.01,n_iter=3000,X=Xdata,y=Ydata)
stopImplicitCluster()

#testing foreach
x <- iris[which(iris[,5] != "setosa"), c(1,5)]
trials <- 10
system.time({
  r <- foreach(icount(trials), .combine=rbind) %dopar% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
  }
})