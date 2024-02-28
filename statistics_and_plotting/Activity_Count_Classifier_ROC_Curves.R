# Protocol 2, Device Sleep Vs. Wake Relative to PSG
library(data.table)
library(ggplot2)
library(ggpubr)
library(pROC)
library(caret)
library(ggpubr)
library(doBy)

figure_output_path <- '../figures/'

base_font_size <- 8
legend_font_size <- 10

output_parameters <- expand.grid(label = c('Penn_State', 'MESA'),
                                 model_prefix = c('24_Hour_Bidirectional_09_09_2023T23_37_28'))


for (r in 1:nrow(output_parameters)){
  
  parameters <- output_parameters[r,]
  label <- parameters$label
  model_prefix <- parameters$model_prefix

  combined_roc_means <- list()
  
  if (label == 'Penn_State'){
    interval_mapping <- c('24_hour' = '~24-Hour Interval',
                          'lights_off' = 'In-Bed Interval')
    title_text <- "~24-Hour Classifier ROC"
    participant_id_format <- '%s'
  } else {
    interval_mapping <- c('24_hour' = 'All Available Interval',
                          'lights_off' = 'In-Bed Interval')
    title_text <- "~24-Hour Classifier on MESA ROC"
    participant_id_format <- '%04i'
    
  }
  
  for (interval in c('24_hour', 'lights_off')){
    data <- read.csv(sprintf('../prediction_summary/%s/%s_%s_%s_by_record.csv', model_prefix, model_prefix, label, interval))
    roc_curve_directory <- sprintf('../prediction_summary/%s/roc_curve_data/%s_%s/', model_prefix, label, interval)
    participant_id_formatted <- sprintf(participant_id_format, data$participant_id)
    roc_curve_filenames <- paste(label, interval, participant_id_formatted, data$participant_session, 'auc_values.csv', sep='_')
    roc_curve_data <- lapply(paste0(roc_curve_directory, roc_curve_filenames), read.csv)
    res <- do.call(rbind, roc_curve_data)
    overall_roc_mean <- summaryBy(formula = fpr + tpr ~ thresholds, data = res, FUN = c(mean, sd))
    overall_roc_mean$interval <- interval
    combined_roc_means[[length(combined_roc_means)+1]] <- overall_roc_mean
  }
  
  roc_mean_df <- do.call(rbind, combined_roc_means)
  roc_mean_df$interval <- interval_mapping[roc_mean_df$interval]
  
  # load all the ROC curves for this data group
  roc_plot_grob <- ggplot(data = roc_mean_df, aes(x = fpr.mean, y = tpr.mean, group = interval, linetype= interval)) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) + 
    geom_line(size = 1) + 
    ggtitle(title_text) + 
    xlab('False Positive Rate (1 - Specificity)') +
    ylab('True Positive Reate (Sensitivity)') + 
    theme_bw(base_size = base_font_size) + 
    theme(plot.title = element_text(face='bold', hjust = 0.5),
          text=element_text(color='black'),
          legend.text=element_text(size=rel(1)),
          axis.text = element_text(size = rel(1), color='black'),
          legend.position = c(.7, .2)) + 
    labs(linetype='Evaluation Interval') +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="grey92", linetype="solid", size = rel(0.5))
  
  ggsave(plot = roc_plot_grob, filename = paste(paste0(figure_output_path, 'pdf/'), 'Figure_X', '_', model_prefix, '_', label, '_5-Fold_CV_ROC.pdf', sep = ''), units = 'mm', width = 89, height = 80, dpi = 300, bg='white')
  ggsave(plot = roc_plot_grob, filename = paste(paste0(figure_output_path, 'png/'), 'Figure_X', '_', model_prefix, '_', label, '_5-Fold_CV_ROC.png', sep = ''), units = 'mm', width = 89, height = 80, dpi = 300, bg='white')
  
}

