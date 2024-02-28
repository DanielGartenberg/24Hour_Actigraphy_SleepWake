# Bland Altman Comparisons
library(BlandAltmanLeh)
library(ggExtra)
library(ggpubr)
library(caret)
library(htmlTable)
library(dplyr)
library(lme4)
library(lmerTest)

debugSource('sleep_metric_comparison_plot.r')

figure_output_path <- '../figures/'

# generate a grid of value combinations to export
output_parameters <- expand.grid(label = c('Penn_State', 'MESA'),
                                 model_prefix = c('24_Hour_Bidirectional_09_09_2023T23_37_28'),
                                 interval = c('24_hour', 'lights_off'))


full_color_map <- c("Sound Sleeping" = '#ff7f00',
                    'Deep Sleeping' = "#377eb8",
                    'Sleep Restriction (Baseline)' = "black",
                    'Sleep Restriction (Restriction)' = '#e41a1c',
                    'Sleep Restriction (Recovery)' = 'gray50',
                    'EcoSleep' = "#984ea3",
                    "MESA" = '#4daf4a')

device_label_map <- c('classifier' = 'TCN',
                      'spectrum' = 'Oakley (mfr. algorithm)')

metric_label_map <- c('se' = 'Sleep Efficiency (%)',
                      'sol' = 'Sleep Onset Latency AASM (Minutes)',
                      'waso' = 'WASO (Minutes)',
                      'tst' = 'Total Sleep Time (Minutes)')

dataset_label_map <- c('SoundSleeping' = 'Sound Sleeping',
                       'DeepSleeping' = 'Deep Sleeping',
                       'Ecosleep' = 'EcoSleep',
                       'sr_baseline' = 'Sleep Restriction (Baseline)',
                       'sr_sr' = 'Sleep Restriction (Restriction)',
                       'sr_recovery' = 'Sleep Restriction (Recovery)',
                       'MESA-COMMERCIAL-USE' = 'MESA')


for (r in 1:nrow(output_parameters)){
  
  parameters <- output_parameters[r,]
  label <- parameters$label
  interval <- parameters$interval
  model_prefix <- parameters$model_prefix
  
  print(label)
  
  if (label == 'Penn_State'){
    color_map_device <- full_color_map[names(full_color_map) != 'MESA']
    se_tick_interval <- 20
    waso_sol_tick_interval <- 60
    full_day_text <- "Statistics Within ~24-Hour Interval"
    loa_type <- 'mixed'
    bias_text_position_24hr = 'top'
    bias_text_position_in_bed = 'bottom'
  } else {
    color_map_device <- full_color_map['MESA']
    se_tick_interval <- 25
    waso_sol_tick_interval <- 120
    full_day_text <- "Statistics Within All Available Data"
    loa_type <- 'non_mixed'
    bias_text_position_24hr = 'bottom'
    bias_text_position_in_bed = 'bottom'
  }
  
  
  # swap in more meaningful labels
  filename <- paste0('../prediction_summary/', model_prefix, '/', model_prefix, '_', label, '_', interval, '_by_record.csv')
  overall_data = read.csv(filename)
  
  # In the MESA data set, the classifier failed to detect any sleep in a subset of records
  # This leads WASO and SOL to be undefined (NA)
  # For NA WASO / SOL values returned by classifier, also set NA in spectrum record, in case these values are not missing at random
  overall_data$spectrum_waso[is.na(overall_data$classifier_waso)] <- NA
  overall_data$spectrum_sol[is.na(overall_data$classifier_sol)] <- NA
  
  # relabel Sleep Restriction datasets by condition
  sleep_restriction_rows <- overall_data$data_set == 'SleepRestriction'
  sr_baseline_rows <- sleep_restriction_rows & grepl('BL0', overall_data$csv_filename)
  sr_sr_rows <- sleep_restriction_rows & grepl('SR0', overall_data$csv_filename)
  sr_rec_rows <- sleep_restriction_rows & grepl('REC0', overall_data$csv_filename)
  
  overall_data$data_set[sr_baseline_rows] <- 'sr_baseline'
  overall_data$data_set[sr_sr_rows] <- 'sr_sr'
  overall_data$data_set[sr_rec_rows] <- 'sr_recovery'
  
  
  # rearrange data from wide to long format
  true_df <- overall_data[,c('data_set', 'participant_id', 'true_sol', 'true_waso', 'true_tst','true_se')]
  # split and label
  classifier_df <- overall_data[,c('classifier_sol', 'classifier_waso', 'classifier_tst', 'classifier_se')]
  spectrum_df <- overall_data[,c('spectrum_sol', 'spectrum_waso', 'spectrum_tst', 'spectrum_se')]
  names(classifier_df) <- sapply(strsplit(names(classifier_df), '_'), function(x) x[2])
  names(spectrum_df) <- sapply(strsplit(names(spectrum_df), '_'), function(x) x[2])
  
  classifier_df <- cbind(classifier_df, true_df)
  classifier_df$device <- 'classifier'
  
  spectrum_df <- cbind(spectrum_df, true_df)
  spectrum_df$device <- 'spectrum'
  
  combined_df <- rbind(classifier_df, spectrum_df)
  
  # relabel the device strings
  combined_df$device <- device_label_map[combined_df$device]
  combined_df$device <- as.factor(combined_df$device)
  
  combined_df$data_set <- dataset_label_map[combined_df$data_set]
  combined_df$data_set = as.factor(combined_df$data_set)
  
  if (interval == 'lights_off'){
    
    sleep_efficiency_grob <- sleep_metric_comparison_plot(combined_df, 'true_se', 'se', 'Classification', color_map_device, 'Sleep Efficiency', '%', 20)
    sleep_onset_grob <- sleep_metric_comparison_plot(combined_df, 'true_sol', 'sol', 'Classification', color_map_device, 'SOL', 'Minutes', waso_sol_tick_interval)
    waso_grob <- sleep_metric_comparison_plot(combined_df, 'true_waso', 'waso', 'Classification', color_map_device, 'WASO', 'Minutes', waso_sol_tick_interval)
    total_sleep_time_grob <- sleep_metric_comparison_plot(combined_df, 'true_tst', 'tst', 'Classification', color_map_device, 'TST', 'Minutes', 120)
    
    sleep_efficiency_difference_grob <- sleep_metric_bland_altman_plot(combined_df, 'true_se', 'se', loa_type, 'Classification', color_map_device, bias_text_position_in_bed, 'Sleep Efficiency', '%', se_tick_interval, 0, 100)
    sleep_onset_difference_grob <- sleep_metric_bland_altman_plot(combined_df, 'true_sol', 'sol', loa_type, 'Classification', color_map_device, bias_text_position_in_bed, 'SOL', 'Minutes', waso_sol_tick_interval, 0)
    waso_difference_grob <- sleep_metric_bland_altman_plot(combined_df, 'true_waso', 'waso', loa_type, 'Classification', color_map_device, bias_text_position_in_bed, 'WASO', 'Minutes', waso_sol_tick_interval, 0)
    total_sleep_time_difference_grob <- sleep_metric_bland_altman_plot(combined_df, 'true_tst', 'tst', loa_type, 'Classification', color_map_device, bias_text_position_in_bed, 'TST', 'Minutes', 120, 0)
    
    full_page_plot <- ggarrange(sleep_onset_grob, sleep_onset_difference_grob,
                                total_sleep_time_grob, total_sleep_time_difference_grob,
                                sleep_efficiency_grob, sleep_efficiency_difference_grob,
                                waso_grob, waso_difference_grob,
                                ncol=2, nrow=4, common.legend=TRUE, legend='bottom') + 
      theme(plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"))
    
    full_page_plot <- annotate_figure(full_page_plot, top = text_grob("Statistics Within In-Bed Interval"))
    
    ggsave(plot = full_page_plot, filename = paste(paste0(figure_output_path, 'pdf/'), 'Figure_X_', model_prefix, '_', label, '_', interval, '_Spectrum_and_Classifier_Full_Filled.pdf', sep = ''), units = 'mm', width = 183, height = 247, dpi = 300, bg='white')
    ggsave(plot = full_page_plot, filename = paste(paste0(figure_output_path, 'png/'), 'Figure_X_', model_prefix, '_', label, '_', interval, '_Spectrum_and_Classifier_Full_Filled.png', sep = ''), units = 'mm', width = 183, height = 247, dpi = 300, bg='white')
    
    
  } else if (interval == '24_hour'){
    
    total_sleep_time_grob <- sleep_metric_comparison_plot(combined_df, 'true_tst', 'tst', 'Classification', color_map_device, 'TST', 'Minutes', 120)
    total_sleep_time_difference_grob <- sleep_metric_bland_altman_plot(combined_df, 'true_tst', 'tst', loa_type, 'Classification', color_map_device, bias_text_position_24hr, 'TST', 'Minutes', 120, 0)
    
    # try the two side by side
    combined_grob <- ggarrange(total_sleep_time_grob, total_sleep_time_difference_grob, ncol=2, nrow=1, common.legend=TRUE, legend='bottom') +
      theme(plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"))
    
    combined_grob <- annotate_figure(combined_grob, top = text_grob(full_day_text))
    
    ggsave(plot = combined_grob, filename = paste(paste0(figure_output_path, 'pdf/'), 'Figure_X_', model_prefix, '_', label, '_', interval, '_Spectrum_and_Classifier_Combined_Filled.pdf', sep = ''), units = 'mm', width = 183, height = 80, dpi = 300, bg='white')
    ggsave(plot = combined_grob, filename = paste(paste0(figure_output_path, 'png/'), 'Figure_X_', model_prefix, '_', label, '_', interval, '_Spectrum_and_Classifier_Combined_Filled.png', sep = ''), units = 'mm', width = 183, height = 80, dpi = 300, bg='white')
    
    
  } 
}



