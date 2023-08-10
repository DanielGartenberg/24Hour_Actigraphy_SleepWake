source('summary_string_utilities.R')
library(lme4)
library(lmerTest)
library(cccrm)
library(epiR)

# function to bootstrap 95% CI for CCC differences (for non-nested data)
# assumes that the data contains three columns representing the true value,
# comparison values, and classifier values, respectively
ccc_difference_epi.ccc <- function(data, indices) {
  d <- data[indices,] # allows boot to select sample
  ccc_true_one <- epi.ccc(d[,1], d[,2])$rho.c$est
  ccc_true_two <- epi.ccc(d[,1], d[,3])$rho.c$est
  ccc_diff <- ccc_true_one - ccc_true_two
  return(ccc_diff)
}

# function to bootstrap 95% CI for CCC differences (for non-nested data)
# assumes that the data contains three columns representing the true value, 
# comparison value one, and comparison value two, respectively
ccc_difference_cccUst <- function(data, indices) {
  d <- data[indices,] # allows boot to select sample
  ccc_true_one <- cccUst_wide(d[,1], d[,2])['CCC']
  ccc_true_two <- cccUst_wide(d[,1], d[,3])['CCC']
  ccc_diff <- ccc_true_one - ccc_true_two
  return(ccc_diff)
}


# function to bootstrap 95% CI for CCC differences (for nested / clustered data)
# assumes the data arrives as a list of data frames
# assumes that the data contains five columns representing the true value, 
# comparison value one, comparison value two, participant identifier, and
# session identifier, respectively
ccc_difference_cluster <- function(data, indices) {
  d <- data[indices] # allows boot to select sample
  # rename 'participant' column of each df to be unique
  for (p in 1:length(d)){
    d[[p]]$participant_id <- paste0('Boot_Participant_', p)
  }
  # reassemble list of dataframes back into one
  d <- do.call('rbind', d)
  
  ccc_true_one <- ccclon_wide(d[,1], d[,2], d[,4], d[,5])['CCC']
  ccc_true_two <- ccclon_wide(d[,1], d[,3], d[,4], d[,5])['CCC']
  ccc_diff <- ccc_true_one - ccc_true_two
  return(ccc_diff)
}


# compute the ccUst with wide data as input, for use in bootstrapping
cccUst_wide <- function(method_one, method_two){
  
  # assemble long dataframe for ccUst
  one_df <- data.frame(method = 'one',
                       outcome = method_one)
  
  two_df <- data.frame(method = 'two',
                       outcome = method_two)
  
  cccUst_df <- rbind(one_df, two_df)
  cccUst_result <- cccUst(cccUst_df, 'outcome', 'method')
  return(cccUst_result)
}

# compute the ccclon with wide data as input, for use in bootstraping
ccclon_wide <- function(method_one, method_two, participant_id, participant_session){
  
  # assemble long dataframe for ccclon
  one_df <- data.frame(method = 'one',
                       outcome = method_one,
                       participant_id = participant_id,
                       participant_session = participant_session)
  
  two_df <- data.frame(method = 'two',
                       outcome = method_two,
                       participant_id = participant_id,
                       participant_session = participant_session)
  
  ccclon_df <- rbind(one_df, two_df)
  ccclon_df[, 'outcome'] <- (ccclon_df[, 'outcome'] - mean(ccclon_df[, 'outcome']) ) / sd(ccclon_df[, 'outcome'])
  # computation of ccclon may fail if the underlying mixed-effects model can't be fit
  ccc <- tryCatch(
    {
      ccclon(ccclon_df, 'outcome', 'participant_id', 'participant_session', 'method')$ccc
    },
    error=function(err_msg){
      message("Fitting ccclon resulted in error, returning NA")
      print(err_msg)
      setNames(array(NA, 6),c("CCC", "LL CI 95%", "UL CI 95%", "SE CCC", "Z", "SE Z"))
    }
  )
  
  return(ccc)
}


# rename SR sessions to the order in which they occurred for mixed-effects CCC
sr_study_day_map <- c("BL01" = "1",
                      "BL02" = "2",
                      "BL03" = "3",
                      "BL04" = "4",
                      "SR01" = "5",
                      "SR02" = "6",
                      "SR03" = "7",
                      "SR04" = "8",
                      "SR05" = "9",
                      "REC01" = "10",
                      "REC02" = "11")


model_names <- c('24_Hour_Bidirectional_09_08_2022T04_59_52', '24_Hour_08_28_2022T05_10_00')

number_bootstrap_resamples <- 10000
bootstrap_parallel <- 'multicore' # Mac and Linux only
bootstrap_ncpus <- 6 # change according to machine, 128 was used on Linux server

comparison_type <- c('spectrum', 'sadeh', 'scripps')

for (model_name in model_names){
  
  output_dir_path <- sprintf('../tables/raw/%s/', model_name)
  if (!dir.exists(output_dir_path)){
    dir.create(output_dir_path)
  }
  
  model_summary_path = sprintf('../prediction_summary/%s/', model_name)
  
  for (label in c('Penn_State', 'MESA')){
    
    for (interval in c('24_hour', 'lights_off')){
      
      for (ct in comparison_type) {
        
        data <- read.csv(sprintf('%s%s_%s_%s_by_record.csv', model_summary_path, model_name, label, interval))
        
        # In the MESA data set, the classifier failed to detect any sleep in a subset of records
        # This leads WASO and SOL to be undefined (NA)
        # For NA WASO / SOL values returned by classifier, also set NA in comparison record, in case these values are not missing at random
        data[, sprintf('%s_waso', ct)][is.na(data$classifier_waso)] <- NA
        data[, sprintf('%s_sol', ct)][is.na(data$classifier_sol)] <- NA
        
        # rename SR participant sessions by the order in which they occurred for mixed-effects CCC
        data$participant_session <- sapply(data$participant_session, function(x) {ifelse(x %in% names(sr_study_day_map), sr_study_day_map[x], x)})
        
        # create versions representing the error between predicted and true
        data$comparison_se_diff <- data[, sprintf('%s_se', ct)] - data$true_se
        data$comparison_sol_diff <- data[, sprintf('%s_sol', ct)] - data$true_sol
        data$comparison_waso_diff <- data[, sprintf('%s_waso', ct)] - data$true_waso
        data$comparison_tst_diff <- data[, sprintf('%s_tst', ct)] - data$true_tst
        
        data$classifier_se_diff <- data$classifier_se - data$true_se
        data$classifier_sol_diff <- data$classifier_sol - data$true_sol
        data$classifier_waso_diff <- data$classifier_waso - data$true_waso
        data$classifier_tst_diff <- data$classifier_tst - data$true_tst
        
        data$comparison_se_absdiff <- abs(data$comparison_se_diff)
        data$comparison_sol_absdiff <- abs(data$comparison_sol_diff)
        data$comparison_waso_absdiff <-  abs(data$comparison_waso_diff)
        data$comparison_tst_absdiff <-  abs(data$comparison_tst_diff)
        
        data$classifier_se_absdiff <- abs(data$classifier_se_diff)
        data$classifier_sol_absdiff <-  abs(data$classifier_sol_diff)
        data$classifier_waso_absdiff <-  abs(data$classifier_waso_diff)
        data$classifier_tst_absdiff <-  abs(data$classifier_tst_diff)
      
        data_source_names <- c('classifier', 'comparison')
        
        if (interval == 'lights_off'){
          metric_names = c('se', 'sol', 'waso', 'tst')
        } else {
          metric_names <- c('tst')
        }
        
        # only the data from Penn State experiments is nested
        if (label == 'Penn_State'){
          ccc_measures = c('CCC', 'CCC_Mixed')
        } else {
          ccc_measures = c('CCC')
        }
        
        # how each metric will be compared
        diff_metric_measures <- c('diff', 'absdiff')
        
        diff_output_df <- expand.grid(metric = metric_names,
                                      measure = diff_metric_measures,
                                      comparison_pooled = NA,
                                      TCN_pooled = NA,
                                      comparison = NA,
                                      TCN = NA,
                                      condition_difference = NA,
                                      DF = NA,
                                      tvalue = NA,
                                      pvalue = NA,
                                      d = NA,
                                      random_effects = NA)
        
        
        ccc_output_df <- expand.grid(metric = metric_names,
                                     measure = ccc_measures,
                                     comparison = NA,
                                     TCN = NA,
                                     difference = NA,
                                     Difference_CI_Excludes_Zero = NA,
                                     num_na_bootstrap_samples = NA)
        
        
        # reshape the differences and abs differences to long format for lmer
        long_format_metric_names = apply(expand.grid(metric_names, diff_metric_measures), 1, paste, collapse="_")
        
        varying_columns = list()
        for (v in long_format_metric_names){
          varying_values <- c()
          for (t in data_source_names){
            varying_values <- append(varying_values, paste(t, v, sep = '_'))
          }
          varying_columns <- append(varying_columns, list(varying_values))
        }
        
        id_names <- c('participant_id', 'participant_session')
        
        # drop the unneeded columns to avoid confusion post-reshape
        columns_to_drop <- names(data)[!is.element(names(data), c(unlist(varying_columns), id_names))]
        
        difference_df <- reshape(data,
                                 direction='long',
                                 drop=columns_to_drop,
                                 varying=varying_columns, 
                                 timevar='type',
                                 times=data_source_names,
                                 v.names=long_format_metric_names,
                                 idvar=id_names)
        
        # numeric version of type to compute Cohen's d with the appropriate contrasts
        type_numeric <- as.numeric(difference_df$type == 'classifier')
        type_numeric[type_numeric == 1] = 1
        type_numeric[type_numeric == 0] = -1
        difference_df$type_numeric <- type_numeric
        
        for (m in metric_names){
          
          print(paste('Processing', model_name, label, interval, ct, m))
        
          for (k in diff_metric_measures){
            
            output_row <- diff_output_df$metric == m & diff_output_df$measure == k
            
            metric_name_meaure <- paste(m, k, sep='_')
            
            if (label == 'Penn_State'){
              lmer_formula <- as.formula(sprintf('%s ~ 1 + type_numeric + (1 + type_numeric | participant_id)', metric_name_meaure))
            } else {
              lmer_formula <- as.formula(sprintf('%s ~ 1 + type_numeric + (1 | participant_id)', metric_name_meaure))
            }
            lmer_result <- lmer(lmer_formula, data = difference_df)
            # if the model with random slopes produces a singular fit, re-run without the random slopes
            if (isSingular(lmer_result)){
              lmer_formula <- as.formula(sprintf('%s ~ 1 + type_numeric + (1 | participant_id)', metric_name_meaure))
              lmer_result <- lmer(lmer_formula, data = difference_df)
            }
            
            lmer_summary <- summary(lmer_result)
            df <- sprintf('%0.2f', lmer_summary$coefficients[2,3])
            t_value <- sprintf('%0.2f', lmer_summary$coefficients[2,4])
            p_value <-  lmer_summary$coefficients[2,5]
            if (p_value < .01){
              p_value <- '<.01'
            } else {
              p_value <- sprintf('%0.2f', p_value)
            }
            
            if (length(ranef(lmer_result)$participant_id) == 2){
              random_effects <- 'slope and intercept'
            } else {
              random_effects <- 'intercept'
            }
            
            # coefficient is multiplied by 2 as conditions are coded -1, 1
            # so a distance of two units represents the difference
            condition_difference <- fixef(lmer_result)[2] * 2
            
            # compute Cohen's d, see:
            # Judd et al. (2017)
            # "Experiments with More Than One Random Factor: Designs, Analytic Models, and Statistical Power
            
            variance_df <- as.data.frame(VarCorr(lmer_result))
            numer <- summary(lmer_result)$coef[2] * 2
            denom <- sqrt(sum(variance_df$vcov))
            d_numeric <- abs(numer / denom)
            d <- sprintf('%0.2f', d_numeric)
            
            # mean and standard deviation, pooled across days (or nights) regardless of participant clustering
            column_values <- aggregate(as.formula(sprintf('%s ~ type', metric_name_meaure)), difference_df, function(x) sprintf('%0.2f (%0.2f)', mean(x, na.rm = TRUE), sd(x, na.rm = TRUE)), na.action = na.pass)
            comparison_value_pooled <- column_values[column_values$type == 'comparison', metric_name_meaure]
            tcn_value_pooled <- column_values[column_values$type == 'classifier', metric_name_meaure]
            
            # mean and standard deviation following mixed effects model fitting intercept only
            if (label == 'Penn_State'){
              intercept_only_formula <- as.formula(sprintf('%s ~ 1 + (1 | participant_id)', metric_name_meaure))
              comparison_data = subset(difference_df, difference_df$type == 'comparison')
              tcn_data = subset(difference_df, difference_df$type == 'classifier')
              
              comparison_fixed_text <- summary_string_mixed_effects(comparison_data[,metric_name_meaure], comparison_data[,'participant_id'])
              tcn_fixed_text <- summary_string_mixed_effects(tcn_data[,metric_name_meaure], tcn_data[,'participant_id'])
            } else {
              comparison_fixed_text = NA
              tcn_fixed_text = NA
            }
            
            diff_output_df[output_row,3:12] <- list(comparison_value_pooled,
                                                    tcn_value_pooled,
                                                    comparison_fixed_text,
                                                    tcn_fixed_text,
                                                    condition_difference,
                                                    df,
                                                    t_value,
                                                    p_value,
                                                    d,
                                                    random_effects)
            
          }
          
          
          # concordance correlation coefficients (CCC)
          for (k in ccc_measures){
            
            # pooled CCC (across all days/nights regardless of clustering within participants)
            ccc_output_row <- ccc_output_df$metric == m & ccc_output_df$measure == k
            
            true_column_name <- sprintf('true_%s', m)
            comparison_column_name <- sprintf('%s_%s', ct, m)
            classifier_column_name <- sprintf('classifier_%s', m)
            
            complete_rows <- complete.cases(data[,c(true_column_name, comparison_column_name, classifier_column_name)])
            ccc_data <- data[complete_rows,]
            
            if (k == 'CCC'){
              ccc_comparison <- epi.ccc(ccc_data[,true_column_name], ccc_data[,comparison_column_name])$rho.c
              ccc_classifier <- epi.ccc(ccc_data[,true_column_name], ccc_data[,classifier_column_name])$rho.c

              # bootstrap the 95% CI for the difference
              ccc_boot_data <- ccc_data[,c(true_column_name, comparison_column_name, classifier_column_name)]
              ccc_boot_result <- boot::boot(data = ccc_boot_data, statistic = ccc_difference_epi.ccc, R = number_bootstrap_resamples, parallel = bootstrap_parallel, ncpus = bootstrap_ncpus)
              num_na_bootstrap_samples <- sum(is.na(ccc_boot_result$t))

            } else if (k == 'CCC_Mixed') {
              ccc_comparison <- ccclon_wide(ccc_data[,true_column_name], ccc_data[,comparison_column_name], ccc_data[,'participant_id'], ccc_data[,'participant_session'])
              ccc_classifier <- ccclon_wide(ccc_data[,true_column_name], ccc_data[,classifier_column_name], ccc_data[,'participant_id'], ccc_data[,'participant_session'])
              
              # bootstrap the 95% CI for the difference with a cluster / 'cases' bootstrap
              ccc_boot_data <- ccc_data[,c(true_column_name, comparison_column_name, classifier_column_name, 'participant_id', 'participant_session')]
              # split into a list of data frames per cluster (participant)
              ccc_boot_data_by_cluster <- split(ccc_boot_data, ccc_boot_data$participant_id)
              ccc_boot_result <- boot::boot(data = ccc_boot_data_by_cluster, statistic = ccc_difference_cluster, R = number_bootstrap_resamples, parallel = bootstrap_parallel, ncpus = bootstrap_ncpus)
              num_na_bootstrap_samples <- sum(is.na(ccc_boot_result$t))
              sprintf('num NA bootstrap samples = %i', num_na_bootstrap_samples)
            }
            
            ccc_boot_ci <- boot::boot.ci(ccc_boot_result, conf = 0.95, type = 'bca')
            
            ccc_output_df[ccc_output_row,'comparison'] <- do.call(sprintf, c(fmt = '%0.2f [%0.2f, %0.2f]', as.list(ccc_comparison[1:3])))
            ccc_output_df[ccc_output_row,'TCN'] <- do.call(sprintf, c(fmt = '%0.2f [%0.2f, %0.2f]', as.list(ccc_classifier[1:3])))
            
            ccc_difference_center <- ccc_comparison[1] - ccc_classifier[1]
            ccc_difference_ci_lower <- ccc_boot_ci$bca[4]
            ccc_difference_ci_upper <- ccc_boot_ci$bca[5]
            
            ccc_output_df[ccc_output_row,'difference'] <- c(sprintf('%0.2f [%0.2f, %0.2f]',ccc_difference_center, ccc_difference_ci_lower, ccc_difference_ci_upper))

            ci_excludes_zero <- !((0 >= ccc_difference_ci_lower) & (0 <= ccc_difference_ci_upper))
            if (ci_excludes_zero){
              ci_excludes_zero_symbol <- '*'
            } else {
              ci_excludes_zero_symbol <- ''
            }
            ccc_output_df[ccc_output_row,'Difference_CI_Excludes_Zero'] <- ci_excludes_zero_symbol
            ccc_output_df[ccc_output_row,'num_na_bootstrap_samples'] <- num_na_bootstrap_samples
            
          }
        }
        
        names(diff_output_df)[names(diff_output_df) == 'comparison_pooled'] <- sprintf('%s_pooled', ct)
        names(diff_output_df)[names(diff_output_df) == 'comparison'] <- ct
        
        names(ccc_output_df)[names(ccc_output_df) == 'comparison'] <- ct
      
        diff_output_df <- format(diff_output_df, digits=1, nsmall=2)
        write.csv(diff_output_df, sprintf('%s/%s_%s_%s_TCN_vs_%s_Night_Level_Differences_Output_Summary.csv', output_dir_path, model_name, label, interval, ct), row.names = FALSE)
        write.csv(ccc_output_df, sprintf('%s/%s_%s_%s_TCN_vs_%s_Night_Level_CCC_Output_Summary.csv', output_dir_path, model_name, label, interval, ct), row.names = FALSE)
        

      }
    }
  }
}

