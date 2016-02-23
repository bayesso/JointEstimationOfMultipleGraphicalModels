
## trainX: data
## trainY: labels for categories (1, 2, 3,...)
## lambda_value: tuning parameter
## Make sure loading "glasso" package before using the code. 

load(glasso)

CGM_AHP_train <- function(trainX, trainY, lambda_value, adaptive_weight=array(1, c(length(unique(Y)), ncol(X), ncol(X))))
{
    ## Set the general paramters
    K <- length(unique(trainY))
    p <- ncol(trainX)
    diff_value <- 1e+10
    count <- 0
    tol_value <- 1e-2
    max_iter <- 30
    
    ## Set the optimizaiton parameters
    OMEGA <- array(0, c(K, p, p))
    S <- array(0, c(K, p, p))
    OMEGA_new <- array(0, c(K, p, p))
    nk <- rep(0, K)
    
    ## Initialize Omega
    for (k in seq(1, K))
    {   
        idx <- which(trainY == k)
        S[k, , ] <- cov(trainX[idx, ]) 
        if (kappa(S[k, , ]) > 1e+15)
        {   
            S[k, , ] <- S[k, , ] + 0.001*diag(p) 
        }
        tmp <- solve(S[k, , ])
        OMEGA[k, , ] <- tmp
        nk[k] <- length(idx)
    }
    
    ## Start loop
    while((count < max_iter) & (diff_value > tol_value))
    {
        tmp <- apply(abs(OMEGA), c(2,3), sum)
        tmp[abs(tmp) < 1e-10] <- 1e-10
        V <- 1 / sqrt(tmp)
        
        for (k in seq(1, K))
        {
            penalty_matrix <- lambda_value * adaptive_weight[k, , ] * V
            obj_glasso <- glasso(S[k, , ], penalty_matrix, maxit=100)
            OMEGA_new[k, , ] <- (obj_glasso$wi + t(obj_glasso$wi)) / 2
            #OMEGA_new[k, , ] <- obj_glasso$wi
        }
        
        ## Check the convergence
        diff_value <- sum(abs(OMEGA_new - OMEGA)) / sum(abs(OMEGA))
        count <- count + 1
        OMEGA <- OMEGA_new
        #cat(count, ', diff_value=', diff_value, '\n')
    }    
    
    ## Filter the noise
    for (k in seq(1, K))
    {
        ome <- OMEGA[k, , ]
        ww <- diag(ome)
        ww[abs(ww) < 1e-10] <- 1e-10
        ww <- diag(1/sqrt(ww))
        tmp <- ww %*% ome %*% ww
        ome[abs(tmp) < 1e-5] <- 0
        OMEGA[k, , ] <- ome
    }
        
    output <- list()
    output$OMEGA <- OMEGA
    output$S <- S
    output$lambda <- lambda_value
        
    return(output)
}