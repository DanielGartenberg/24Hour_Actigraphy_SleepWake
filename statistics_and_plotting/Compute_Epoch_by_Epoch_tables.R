source('summary_string_utilities.R')
library(lme4)
library(lmerTest)

model_names <- c('24_Hour_09_10_2023T18_38_25', '24_Hour_Bidirectional_09_09_2023T23_37_28')

comparison_type <- c('spectrum', 'sadeh', 'scripps')

for (model_name in model_names){
  
  model_summary_path <- sprintf('../prediction_summary/%s/', model_name)
  
  output_dir_path <- sprintf('../tables/raw/%s/', model_name)
  if (!dir.exists(output_dir_path)){
    dir.create(output_dir_path)
  }
  
  for (label in c('Penn_State', 'MESA')){
    
    for (interval in c('24_hour', 'lights_off')){
      
      for (ct in comparison_type) {
        
        data <- read.csv(sprintf('%s%s_%s_%s_by_record.csv', model_summary_path, model_name, label, interval))
        
        time_names <- c('classifier', ct)
        varying_names <- c('auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'mcc', 'pabak')
        
        output_df <- data.frame(metric = varying_names,
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
        
        varying_columns = list()
        for (vn in varying_names){
          varying_values <- c()
          for (tn in time_names){
            varying_values <- append(varying_values, paste(tn, vn, sep = '_'))
          }
          varying_columns <- append(varying_columns, list(varying_values))
        }
        
        id_names <- c('participant_id', 'participant_session')
        
        # drop the unneeded columns to avoid confusion post-reshape
        columns_to_drop <- names(data)[!is.element(names(data), c(unlist(varying_columns), id_names))]
        
        reshaped_epoch_metrics <- reshape(data,
                                          direction='long',
                                          drop=columns_to_drop,
                                          varying=varying_columns, 
                                          timevar='type',
                                          times=time_names,
                                          v.names=varying_names,
                                          idvar=id_names)
        
        # numeric version of type to compute Cohen's d with the appropriate contrasts
        type_numeric <- as.numeric(reshaped_epoch_metrics$type == 'classifier')
        type_numeric[type_numeric == 1] = 1
        type_numeric[type_numeric == 0] = -1
        reshaped_epoch_metrics$type_numeric <- type_numeric
        
        for (m in varying_names){
          
          print(m)
          
          output_row <- output_df$metric == m
          
          if (m != 'auc'){
            if (label == 'Penn_State'){
              lmer_formula <- as.formula(sprintf('%s ~ 1 + type_numeric + (1 + type_numeric | participant_id)', m))
            } else {
              lmer_formula <- as.formula(sprintf('%s ~ 1 + type_numeric + (1 | participant_id)', m))
            }
            lmer_result <- lmer(lmer_formula, data = reshaped_epoch_metrics)
            # if the model with random slopes produces a singular fit, re-run without the random slopes
            if (isSingular(lmer_result)){
              lmer_formula <- as.formula(sprintf('%s ~ 1 + type_numeric + (1 | participant_id)', m))
              lmer_result <- lmer(lmer_formula, data = reshaped_epoch_metrics)
            }
            
            lmer_summary <- summary(lmer_result)
            df <- sprintf('%0.2f', lmer_summary$coefficients[2,3])
            t_value <- sprintf('%0.2f', lmer_summary$coefficients[2,4])
            p_value <-  lmer_summary$coefficients[2,5]
            if (p_value < .01){
              p_value <- '<.01'
            } else {
              p_value <- sprintf('%0.3f', p_value)
            }
            
            if (length(ranef(lmer_result)$participant_id) == 2){
              random_effects <- 'slope and intercept'
            } else {
              random_effects <- 'intercept'
            }
            
            # coefficient is multiplied by 2 as conditions are coded -1, 1
            # so a distance of two units represents the difference
            condition_difference <- sprintf('%0.3f', fixef(lmer_result)[2] * 2)
            
            # compute Cohen's d, see:
            # Judd et al. (2017)
            # "Experiments with More Than One Random Factor: Designs, Analytic Models, and Statistical Power
            
            variance_df <- as.data.frame(VarCorr(lmer_result))
            numer <- summary(lmer_result)$coef[2] * 2
            denom <- sqrt(sum(variance_df$vcov))
            d_numeric <- abs(numer / denom)
            d <- sprintf('%0.2f', d_numeric)
            
            
          # AUC values can't be computed from the discreteclassifiers and thus can't be compared
          } else {
            df <- NA
            t_value  <- NA
            p_value <- NA
            condition_difference <- NA
            d <- NA
            random_effects <- 'NA'
          }
          
          column_values <- aggregate(as.formula(sprintf('%s ~ type', m)), reshaped_epoch_metrics, function(x) sprintf('%0.3f (%0.3f)', mean(x, na.rm = TRUE), sd(x)), na.action = na.pass)
          comparison_value <- column_values[column_values$type == ct, m]
          tcn_value <- column_values[column_values$type == 'classifier', m]
          
          # mean and standard deviation following mixed effects model fitting intercept only
          if (label == 'Penn_State'){
            
            if (m != 'auc'){
              comparison_data = reshaped_epoch_metrics[reshaped_epoch_metrics$type == ct, ]
              comparison_fixed_text <- summary_string_mixed_effects(comparison_data[,m], comparison_data[,'participant_id'], digits=3)
            } else {
              comparison_fixed_text <- 'NA'
            }
            
            tcn_data = reshaped_epoch_metrics[reshaped_epoch_metrics$type == 'classifier', ]
            tcn_fixed_text <- summary_string_mixed_effects(tcn_data[,m], tcn_data[,'participant_id'], digits=3)
            
          } else {
            comparison_fixed_text = NA
            tcn_fixed_text = NA
          }
          
          output_df[output_row,2:11] <- list(comparison_value, tcn_value,
                                             comparison_fixed_text, tcn_fixed_text,
                                             condition_difference, df, t_value,
                                             p_value, d, random_effects)
          
        }
        
        names(output_df)[names(output_df) == 'comparison_pooled'] <- sprintf('%s_pooled', ct)
        names(output_df)[names(output_df) == 'comparison'] <- ct
        
        write.csv(output_df, sprintf('%s/%s_%s_%s_TCN_vs_%s_Epoch_Output_Summary.csv', output_dir_path, model_name, label, interval, ct), row.names = FALSE)
        
      }  
    }
  }
}

