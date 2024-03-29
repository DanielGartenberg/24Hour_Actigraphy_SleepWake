library(boot)
library(lme4)
library(lmerTest)
library(SimplyAgree)


# Limits of Agreement calculation for non-nested data extracted
# from the 'BAplot' function from: https://github.com/SRI-human-sleep/sleep-trackers-performance
loa_non_mixed <- function(data=NA,measures=c("TST_device","TST_ref"),logTransf=FALSE,
                   xaxis="reference",CI.type="classic",CI.level=.95,boot.type="basic",boot.R=10000,
                   xlim=NA,ylim=NA,warnings=TRUE){
  
  require(BlandAltmanLeh); require(ggplot2); require(ggExtra)
  
  
  # setting labels
  Measure <- gsub("_ref","",gsub("_device","",measures[1]))
  measure <- gsub("TST","total sleep time (min)",Measure)
  measure <- gsub("SE","sleep efficiency (%)",measure)
  measure <- gsub("SOL","sleep onset latency (min)",measure)
  measure <- gsub("WASO","wake after sleep onset (min)",measure)
  measure <- gsub("LightPerc","light sleep percentage (%)",measure)
  measure <- gsub("DeepPerc","deep sleep percentage (%)",measure)
  measure <- gsub("REMPerc","REM sleep percentage (%)",measure)
  measure <- gsub("Light","light sleep duration (min)",measure)
  measure <- gsub("Deep","deep sleep duration (min)",measure)
  if(grepl("REM",measure) & !grepl("%",measure)) { measure <- gsub("REM","REM sleep duration (min)",measure) }
  if(warnings==TRUE){cat("\n\n----------------\n Measure:",Measure,"\n----------------")}
  
  

  # packages and functions to be used with bootstrap CI
  if(CI.type=="boot"){ require(boot)
    # function to generate bootstrap CI for model parameters
    boot.reg <- function(data,formula,indices){ return(coef(lm(formula,data=data[indices,]))[2]) }
    # function for sampling and predicting Y values based on model
    boot.pred <- function(data,formula,tofit) { indices <- sample(1:nrow(data),replace = TRUE)
    return(predict(lm(formula,data=data[indices,]), newdata=data.frame(tofit))) }
    if(warnings==TRUE){cat("\n\nComputing boostrap CI with method '",boot.type,"' ...",sep="")}
  } else if(CI.type!="classic") { stop("Error: CI.type can be either 'classic' or 'boot'") }
  
  # data to be used
  ba.stat <- bland.altman.stats(data[,measures[1]],data[,measures[2]],conf.int=CI.level)
  if(xaxis=="reference"){
    ba <- data.frame(size=ba.stat$groups$group2,diffs=ba.stat$diffs)
  } else if(xaxis=="mean"){
    ba <- data.frame(size=ba.stat$means,diffs=ba.stat$diffs)
  } else { stop("Error: xaxis argument can be either 'reference' or 'mean'") }
  
  # range of values to be fitted for drawing the lines (i.e., from min to max of x-axis values, by .1)
  size <- seq(min(ba$size),max(ba$size),(max(ba$size)-min(ba$size))/((max(ba$size)-min(ba$size))*10))
  
  # ..........................................
  # 1. TESTING PROPORTIONAL BIAS
  # ..........................................
  m <- lm(diffs~size,ba)
  if(CI.type=="classic"){ CI <- confint(m,level=CI.level)[2,] 
  } else { CI <- boot.ci(boot(data=ba,statistic=boot.reg,formula=diffs~size,R=boot.R),
                         type=boot.type,conf=CI.level)[[4]][4:5] }
  prop.bias <- ifelse(CI[1] > 0 | CI[2] < 0, TRUE, FALSE)
  
  # ...........................................
  # 1.1. DIFFERENCES INDEPENDENT FROM SIZE
  # ...........................................
  if(prop.bias == FALSE){ 
    
    if(CI.type=="boot"){ # changing bias CI when CI.type="boot"
      ba.stat$CI.lines[3] <- boot.ci(boot(ba$diffs,function(dat,idx)mean(dat[idx],na.rm=TRUE),R=boot.R),
                                     type=boot.type,conf=CI.level)[[4]][4]
      ba.stat$CI.lines[4] <- boot.ci(boot(ba$diffs,function(dat,idx)mean(dat[idx],na.rm=TRUE),R=boot.R),
                                     type=boot.type,conf=CI.level)[[4]][5] }
    
    # output the bias, instead of plotting as in original BAplot
    bias_df <- data.frame(ref = size,
                          estimate = ba.stat$mean.diffs,
                          lower.ci = ba.stat$CI.lines[3],
                          upper.ci = ba.stat$CI.lines[4],
                          row.names = NULL)
      
    # ..........................................
    # 1.2. DIFFERENCES PROPORTIONAL TO SIZE
    # ..........................................
  } else {
    
    b0 <- coef(m)[1]
    b1 <- coef(m)[2]
    
    # warning message
    if(warnings==TRUE){cat("\n\nWARNING: differences in ",Measure," might be proportional to the size of measurement (coeff. = ",
                           round(b1,2)," [",round(CI[1],2),", ",round(CI[2],2),"]",").",
                           "\nBias and LOAs are plotted as a function of the size of measurement.",sep="")}
    
    # modeling bias following Bland & Altman (1999): D = b0 + b1 * size
    y.fit <- data.frame(size,y.bias=b0+b1*size) 
    
    # bias CI
    if(CI.type=="classic"){ # classic ci
      y.fit$y.biasCI.upr <- predict(m,newdata=data.frame(y.fit$size),interval="confidence",level=CI.level)[,3]
      y.fit$y.biasCI.lwr <- predict(m,newdata=data.frame(y.fit$size),interval="confidence",level=CI.level)[,2]
    } else { # boostrap CI 
      fitted <- t(replicate(boot.R,boot.pred(ba,"diffs~size",y.fit))) # sampling CIs
      y.fit$y.biasCI.upr <- apply(fitted,2,quantile,probs=c((1-CI.level)/2))
      y.fit$y.biasCI.lwr <- apply(fitted,2,quantile,probs=c(CI.level+(1-CI.level)/2)) }
    
    # output the bias, instead of plotting as was done in BAplot
    bias_df <- data.frame(ref = size,
                          estimate = y.fit$y.bias,
                          lower.ci = y.fit$y.biasCI.lwr,
                          upper.ci = y.fit$y.biasCI.upr,
                          row.names = NULL)
    }
  
  # ..............................................
  # 2. LOAs ESTIMATION FROM ORIGINAL DATA
  # ..............................................
  if(logTransf == FALSE){
    
    # testing heteroscedasticity
    mRes <- lm(abs(resid(m))~size,ba)
    if(CI.type=="classic"){ CIRes <- confint(mRes,level=CI.level)[2,]
    } else { CIRes <- boot.ci(boot(data=ba,statistic=boot.reg,formula=abs(resid(m))~size,R=boot.R),
                              type=boot.type,conf=CI.level)[[4]][4:5] }
    heterosced <- ifelse(CIRes[1] > 0 | CIRes[2] < 0,TRUE,FALSE)
    
    # testing normality of differences
    shapiro <- shapiro.test(ba$diffs)
    if(shapiro$p.value <= .05){
      if(warnings==TRUE){cat("\n\nWARNING: differences in ",Measure,
                             " might be not normally distributed (Shapiro-Wilk W = ",round(shapiro$statistic,3),", p = ",round(shapiro$p.value,3),
                             ").","\nBootstrap CI (CI.type='boot') and log transformation (logTransf=TRUE) are recommended.",sep="")} }
    
    # ............................................
    # 2.1. CONSTANT BIAS AND HOMOSCEDASTICITY
    # ............................................
    if(prop.bias==FALSE & heterosced==FALSE){
      
      if(CI.type=="boot"){ # changing LOAs CI when CI.type="boot"
        ba.stat$CI.lines[1] <- boot.ci(boot(ba$diffs-1.96*sd(ba.stat$diffs),
                                            function(dat,idx)mean(dat[idx],na.rm=TRUE),R=boot.R),
                                       type=boot.type,conf=CI.level)[[4]][4]
        ba.stat$CI.lines[2] <- boot.ci(boot(ba$diffs-1.96*sd(ba.stat$diffs),
                                            function(dat,idx)mean(dat[idx],na.rm=TRUE),R=boot.R),
                                       type=boot.type,conf=CI.level)[[4]][5]
        ba.stat$CI.lines[5] <- boot.ci(boot(ba$diffs+1.96*sd(ba.stat$diffs),
                                            function(dat,idx)mean(dat[idx],na.rm=TRUE),R=boot.R),
                                       type=boot.type,conf=CI.level)[[4]][4]
        ba.stat$CI.lines[6] <- boot.ci(boot(ba$diffs+1.96*sd(ba.stat$diffs),
                                            function(dat,idx)mean(dat[idx],na.rm=TRUE),R=boot.R),
                                       type=boot.type,conf=CI.level)[[4]][5] }
      
      # output the loa, instead of plotting as was done in BAplot
      loa_u_df <- data.frame(ref = size,
                             estimate = ba.stat$upper.limit,
                             lower.ci = ba.stat$CI.lines[5],
                             upper.ci = ba.stat$CI.lines[6],
                             row.names = NULL)
      
      loa_l_df <- data.frame(ref = size,
                             estimate = ba.stat$lower.limit,
                             lower.ci = ba.stat$CI.lines[1],
                             upper.ci = ba.stat$CI.lines[2],
                             row.names = NULL)
      
      # ............................................
      # 2.2. PROPORTIONAL BIAS AND HOMOSCEDASTICITY
      # ............................................
    } else if(prop.bias==TRUE & heterosced==FALSE) { 
      
      # modeling LOAs following Bland & Altman (1999): LOAs = bias +- 1.96sd of the residuals
      y.fit$y.LOAu = b0+b1*size + 1.96*sd(resid(m))
      y.fit$y.LOAl = b0+b1*size - 1.96*sd(resid(m))
      
      # LOAs CI based on bias CI +- 1.96sd of the residuals
      if(warnings==TRUE){cat(" Note that LOAs CI are represented based on bias CI.")}
      y.fit$y.LOAu.upr = y.fit$y.biasCI.upr + 1.96*sd(resid(m))
      y.fit$y.LOAu.lwr = y.fit$y.biasCI.lwr + 1.96*sd(resid(m))
      y.fit$y.LOAl.upr = y.fit$y.biasCI.upr - 1.96*sd(resid(m))
      y.fit$y.LOAl.lwr = y.fit$y.biasCI.lwr - 1.96*sd(resid(m))
      
      # ............................................
      # 2.3. CONSTANT BIAS AND HOMOSCEDASTICITY
      # ............................................
    } else if(prop.bias==FALSE & heterosced==TRUE) {
      
      c0 <- coef(mRes)[1]
      c1 <- coef(mRes)[2]
      
      # warning message
      if(warnings==TRUE){cat("WARNING: SD of differences in ",Measure,
                             " might be proportional to the size of measurement (coeff. = ",
                             round(c1,2)," [",round(CIRes[1],2),", ",round(CIRes[2],2),"]",").",
                             "\nLOAs range is plotted as a function of the size of measurement.",sep="")}
      
      # modeling LOAs following Bland & Altman (1999): LOAs = meanDiff +- 2.46(c0 + c1A)
      y.fit <- data.frame(size=size,
                          y.LOAu = ba.stat$mean.diffs + 2.46*(c0+c1*size),
                          y.LOAl = ba.stat$mean.diffs - 2.46*(c0+c1*size))
      
      # LOAs CI
      if(CI.type=="classic"){ # classic ci
        fitted <- predict(mRes,newdata=data.frame(y.fit$size),interval="confidence",level=CI.level) # based on mRes
        y.fit$y.LOAu.upr <- ba.stat$mean.diffs + 2.46*fitted[,3]
        y.fit$y.LOAu.lwr <- ba.stat$mean.diffs + 2.46*fitted[,2]
        y.fit$y.LOAl.upr <- ba.stat$mean.diffs - 2.46*fitted[,3]
        y.fit$y.LOAl.lwr <- ba.stat$mean.diffs - 2.46*fitted[,2]
      } else { # boostrap CI
        fitted <- t(replicate(boot.R,boot.pred(ba,"abs(resid(lm(diffs ~ size))) ~ size",y.fit)))
        y.fit$y.LOAu.upr <- ba.stat$mean.diffs + 2.46*apply(fitted,2,quantile,probs=c(CI.level+(1-CI.level)/2))
        y.fit$y.LOAu.lwr <- ba.stat$mean.diffs + 2.46*apply(fitted,2,quantile,probs=c((1-CI.level)/2))
        y.fit$y.LOAl.upr <- ba.stat$mean.diffs - 2.46*apply(fitted,2,quantile,probs=c(CI.level+(1-CI.level)/2))
        y.fit$y.LOAl.lwr <- ba.stat$mean.diffs - 2.46*apply(fitted,2,quantile,probs=c((1-CI.level)/2)) }
      
      # ............................................
      # 2.4. PROPORTIONAL BIAS AND HETEROSCEDASTICITY
      # ............................................ 
    } else if(prop.bias==TRUE & heterosced==TRUE) {
      
      c0 <- coef(mRes)[1]
      c1 <- coef(mRes)[2]
      
      # warning message
      if(warnings==TRUE){cat("\n\nWARNING: SD of differences in ",Measure,
                             " might be proportional to the size of measurement (coeff. = ",
                             round(c1,2)," [",round(CIRes[1],2),", ",round(CIRes[2],2),"]",").",
                             "\nLOAs range is plotted as a function of the size of measurement.",sep="")}
      
      # modeling LOAs following Bland & Altman (1999): LOAs = b0 + b1 * size +- 2.46(c0 + c1A) 
      y.fit$y.LOAu = b0+b1*size + 2.46*(c0+c1*size)
      y.fit$y.LOAl = b0+b1*size - 2.46*(c0+c1*size)
      
      # LOAs CI
      if(CI.type=="classic"){ # classic ci
        fitted <- predict(mRes,newdata=data.frame(y.fit$size),interval="confidence",level=CI.level) # based on mRes
        y.fit$y.LOAu.upr <- b0+b1*size + 2.46*fitted[,3]
        y.fit$y.LOAu.lwr <- b0+b1*size + 2.46*fitted[,2]
        y.fit$y.LOAl.upr <- b0+b1*size - 2.46*fitted[,3]
        y.fit$y.LOAl.lwr <- b0+b1*size - 2.46*fitted[,2] 
      } else { # boostrap CI
        fitted <- t(replicate(boot.R,boot.pred(ba,"abs(resid(lm(diffs ~ size))) ~ size",y.fit)))
        y.fit$y.LOAu.upr <- b0+b1*size + 2.46*apply(fitted,2,quantile,probs=c(CI.level+(1-CI.level)/2))
        y.fit$y.LOAu.lwr <- b0+b1*size + 2.46*apply(fitted,2,quantile,probs=c((1-CI.level)/2))
        y.fit$y.LOAl.upr <- b0+b1*size - 2.46*apply(fitted,2,quantile,probs=c(CI.level+(1-CI.level)/2))
        y.fit$y.LOAl.lwr <- b0+b1*size - 2.46*apply(fitted,2,quantile,probs=c((1-CI.level)/2)) }}
    
    if(prop.bias==TRUE | heterosced==TRUE){
      
      # output the loa, instead of plotting as was done in BAplot
      loa_u_df <- data.frame(ref = size,
                             estimate = y.fit$y.LOAu,
                             lower.ci = y.fit$y.LOAu.lwr,
                             upper.ci = y.fit$y.LOAu.upr,
                             row.names = NULL)
      
      loa_l_df <- data.frame(ref = size,
                             estimate = y.fit$y.LOAl,
                             lower.ci = y.fit$y.LOAl.lwr,
                             upper.ci = y.fit$y.LOAl.upr,
                             row.names = NULL)
      
    }
    
    # ..............................................
    # 3. LOAs ESTIMATION FROM LOG-TRANSFORMED DATA
    # ..............................................
  } else {
    
    # log transformation of data (add little constant to avoid Inf values)
    if(warnings==TRUE){cat("\n\nLog transforming data ...")}
    ba.stat$groups$LOGgroup1 <- log(ba.stat$groups$group1 + .0001)
    ba.stat$groups$LOGgroup2 <- log(ba.stat$groups$group2 + .0001)
    ba.stat$groups$LOGdiff <- ba.stat$groups$LOGgroup1 - ba.stat$groups$LOGgroup2
    if(xaxis=="reference"){ baLog <- data.frame(size=ba.stat$groups$LOGgroup2,diffs=ba.stat$groups$LOGdiff)
    } else { baLog <- data.frame(size=(ba.stat$groups$LOGgroup1 + ba.stat$groups$LOGgroup2)/2,diffs=ba.stat$groups$LOGdiff) }
    
    # testing heteroscedasticity
    mRes <- lm(abs(resid(m))~size,baLog)
    if(CI.type=="classic"){ CIRes <- confint(mRes,level=CI.level)[2,]
    } else { CIRes <- boot.ci(boot(data=baLog,statistic=boot.reg,formula=abs(resid(m))~size,R=boot.R),
                              type=boot.type,conf=CI.level)[[4]][4:5] }
    heterosced <- ifelse(CIRes[1] > 0 | CIRes[2] < 0,TRUE,FALSE)
    
    # testing normality of differences
    shapiro <- shapiro.test(baLog$diffs)
    if(shapiro$p.value <= .05){
      if(warnings==TRUE){cat("\n\nWARNING: differences in log transformed ",Measure,
                             " might be not normally distributed (Shapiro-Wilk W = ",round(shapiro$statistic,3),", p = ",round(shapiro$p.value,3),
                             ").","\nBootstrap CI (CI.type='boot') are recommended.",sep="")} }
    
    # LOAs slope following Euser et al (2008) for antilog transformation: slope = 2 * (e^(1.96 SD) - 1)/(e^(1.96 SD) + 1)
    ANTILOGslope <- function(x){ 2 * (exp(1.96 * sd(x)) - 1) / (exp(1.96*sd(x)) + 1) }
    ba.stat$LOA.slope <- ANTILOGslope(baLog$diffs)
    
    # LOAs CI slopes 
    if(CI.type=="classic"){ # classic CI
      t1 <- qt((1 - CI.level)/2, df = ba.stat$based.on - 1) # t-value right
      t2 <- qt((CI.level + 1)/2, df = ba.stat$based.on - 1) # t-value left
      ba.stat$LOA.slope.CI.upper <- 2 * (exp(1.96 * sd(baLog$diffs) + t2 * sqrt(sd(baLog$diffs)^2 * 3/ba.stat$based.on)) - 1) /
        (exp(1.96*sd(baLog$diffs) + t2 * sqrt(sd(baLog$diffs)^2 * 3/ba.stat$based.on)) + 1)
      ba.stat$LOA.slope.CI.lower <- 2 * (exp(1.96 * sd(baLog$diffs) + t1 * sqrt(sd(baLog$diffs)^2 * 3/ba.stat$based.on)) - 1) /
        (exp(1.96*sd(baLog$diffs) + t1 * sqrt(sd(baLog$diffs)^2 * 3/ba.stat$based.on)) + 1)
    } else { # boostrap CI
      ba.stat$LOA.slope.CI.upper <- boot.ci(boot(baLog$diffs,
                                                 function(dat,idx) ANTILOGslope(dat[idx]),R=boot.R),
                                            type=boot.type,conf=CI.level)[[4]][4]
      ba.stat$LOA.slope.CI.lower <- boot.ci(boot(baLog$diffs,
                                                 function(dat,idx) ANTILOGslope(dat[idx]),R=boot.R),
                                            type=boot.type,conf=CI.level)[[4]][5] }
    
    # Recomputing LOAs and their CIs as a function of size multiplied by the computed slopes
    y.fit <- data.frame(size,
                        ANTLOGdiffs.upper = size * ba.stat$LOA.slope, # upper LOA
                        ANTLOGdiffs.upper.lower = size * ba.stat$LOA.slope.CI.lower,
                        ANTLOGdiffs.upper.upper = size * ba.stat$LOA.slope.CI.upper,
                        ANTLOGdiffs.lower = size * ((-1)*ba.stat$LOA.slope), # lower LOA
                        ANTLOGdiffs.lower.lower = size * ((-1)*ba.stat$LOA.slope.CI.lower),
                        ANTLOGdiffs.lower.upper = size * ((-1)*ba.stat$LOA.slope.CI.upper))
    
    # adding bias values based on prop.bias
    if(prop.bias==FALSE){ y.fit$y.bias <- rep(ba.stat$mean.diffs,nrow(y.fit)) } else { y.fit$y.bias <- b0+b1*y.fit$size }
    
    # output the loa, instead of plotting as was done in BAplot
    loa_u_df <- data.frame(ref = size,
                           estimate = y.fit$y.bias + y.fit$ANTLOGdiffs.upper,
                           lower.ci = y.fit$y.bias + y.fit$ANTLOGdiffs.upper.lower,
                           upper.ci = y.fit$y.bias + y.fit$ANTLOGdiffs.upper.upper,
                           row.names = NULL)
    
    loa_l_df <- data.frame(ref = size,
                           estimate = y.fit$y.bias + y.fit$ANTLOGdiffs.lower,
                           lower.ci = y.fit$y.bias + y.fit$ANTLOGdiffs.lower.lower,
                           upper.ci = y.fit$y.bias + y.fit$ANTLOGdiffs.lower.upper,
                           row.names = NULL)
    
    # ..........................................
    # 3.3. HETEROSCHEDASTICITY (only a warning)
    # ..........................................
    if(heterosced==TRUE){
      
      # warning message
      if(warnings==TRUE){cat("\n\nWARNING: standard deviation of differences in in log transformed ",Measure,
                             " might be proportional to the size of measurement (coeff. = ",
                             round(coef(mRes)[2],2)," [",round(CIRes[1],2),", ",round(CIRes[2],2),"]",").",sep="")} }}
  
  
  # reformat to be in same format as loa_mixed
  bias_df$term = 'Bias'
  loa_u_df$term = 'Lower LoA'
  loa_l_df$term = 'Upper LoA'
  
  loa_df <- do.call(rbind, list(bias_df, loa_u_df, loa_l_df))
  
  return(loa_df)  
  
}


# loa for mixed models, using the 'loa_lme' function from the SimplyAgree package
loa_mixed <- function(data_df=NA, difference, psg_var, participant_var){
  
  # limits of agreement for mixed-effects model
  prop_bias_formula <- as.formula(sprintf('%s ~ 1 + %s + (1 | %s)', difference, psg_var, participant_var))
  
  # test whether plotting proportional limits of agreement is warranted
  # fit a mixed-effects model for the relationship between true values and differences
  proportional_bias_test <- lmer(prop_bias_formula, data = data_df)
  if (isSingular(proportional_bias_test)){
    print('Proportional Bias Test Returns Singular Fit')
  }
  proportional_bias_test_summary <- summary(proportional_bias_test)
  significant_prop_bias <- proportional_bias_test_summary$coefficients[2,5] < .05
  
  loa_df <- SimplyAgree::loa_lme(difference, psg_var, NULL, participant_var, data_df, prop_bias = significant_prop_bias, replicates = 10000)$loa
  
  # if there was *not* significant prop bias, replicate the single values
  # so that they can be plotted as a line, similar to when prop bias exists
  if (!significant_prop_bias){
    x_min_df <- loa_df
    x_min_df$avg <- min(data_df[,psg_var])
    x_max_df <- loa_df
    x_max_df$avg <- max(data_df[,psg_var])
    loa_df <- rbind(x_min_df, x_max_df)
  }
  
  # rename for consistency with non-mixed LoA function
  names(loa_df)[names(loa_df) == 'avg'] = 'ref'
  
  return(loa_df)
}


