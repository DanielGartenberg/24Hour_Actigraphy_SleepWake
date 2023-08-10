# utility functions to summarize mean, standard deviation, and 95% CI as strings

library(lme4)

summary_string <- function(sample_values){
  # Compute and format the mean, sample SD, and 95% CI for the mean for a sample
  sammple_n <- length(sample_values)
  sample_mean <- mean(sample_values)
  sample_sd <- sd(sample_values)
  sample_se <- sample_sd / sqrt(sammple_n)
  ci_95_lower <-  sample_mean - (1.96 * sample_se)
  ci_95_upper <-  sample_mean + (1.96 * sample_se)
  return(sprintf('%05.2f (%05.2f) [%05.2f, %05.2f]', sample_mean, sample_sd, ci_95_lower, ci_95_upper))
}


summary_string_mixed_effects <- function(sample_values, identifier, compute_ci = FALSE, confint_nsim=10000, digits=2){
  # Compute and format the mean, SD, and 95% CI for a sample where observations
  # are nested within groups (participants)
  lmer_df <- data.frame(sample_values = sample_values, identifier = identifier)
  intercept_only_model <- lmer(sample_values ~ 1 + (1 | identifier), data = lmer_df)
  model_fixef <- fixef(intercept_only_model)[1]
  model_sds <- as.data.frame(VarCorr(intercept_only_model))$sdcor
  formatted_text <- sprintf('%02.*f (%02.*f; %02.*f)', digits, model_fixef, digits, model_sds[1], digits, model_sds[2])
  if (compute_ci){
    ci <- confint(intercept_only_model, level = 0.95, method = 'boot', nsim = confint_nsim, boot.type = 'perc')
    ci_text <- sprintf('[%02.*f, %02.*f]', digits, ci['(Intercept)',1], digits, ci['(Intercept)',2])
    formatted_text <- paste(formatted_text, ci_text)
  }
  return(formatted_text)
}
