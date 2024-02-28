library(lme4)
library(lmerTest)
source('summary_string_utilities.R')

model_names <- c('24_Hour_09_10_2023T18_38_25', '24_Hour_Bidirectional_09_09_2023T23_37_28')

should_compute_ci = TRUE

for (model_name in model_names){
  
  output_dir_path <- sprintf('../tables/raw/%s/', model_name)
  if (!dir.exists(output_dir_path)){
    dir.create(output_dir_path)
  }
  
  model_summary_path = sprintf('../prediction_summary/%s/', model_name)
  
  for (label in c('Penn_State', 'MESA')){
    
    for (interval in c('24_hour', 'lights_off')){
      
      print(sprintf('Model = %s, Label = %s, interval = %s', model_name, label, interval))
      
      data <- read.csv(sprintf('%s%s_%s_%s_by_record.csv', model_summary_path, model_name, label, interval))
      
      confusion_table <- data.frame('True.Wake' = array(NA, 4), 'True.Sleep' = array(NA, 4),
                                    row.names = c('Spectrum.Predicted.Wake', 'Spectrum.Predicted.Sleep', 'Classifier.Predicted.Wake', 'Classifier.Predicted.Sleep'))
      
      confusion_table['Spectrum.Predicted.Wake', 'True.Wake'] <- summary_string(data$spectrum_TN * 100)
      confusion_table['Spectrum.Predicted.Wake', 'True.Sleep'] <- summary_string(data$spectrum_FN * 100)
      confusion_table['Spectrum.Predicted.Sleep', 'True.Sleep'] <- summary_string(data$spectrum_TP * 100)
      confusion_table['Spectrum.Predicted.Sleep', 'True.Wake'] <- summary_string(data$spectrum_FP * 100)
      
      confusion_table['Classifier.Predicted.Wake', 'True.Wake'] <- summary_string(data$classifier_TN * 100)
      confusion_table['Classifier.Predicted.Wake', 'True.Sleep'] <- summary_string(data$classifier_FN * 100)
      confusion_table['Classifier.Predicted.Sleep', 'True.Sleep'] <- summary_string(data$classifier_TP * 100)
      confusion_table['Classifier.Predicted.Sleep', 'True.Wake'] <- summary_string(data$classifier_FP * 100)
      
      write.csv(confusion_table, sprintf('%s/%s_%s_%s_Confusion_Table.csv', output_dir_path, model_name, label, interval))
      
      if (label == 'Penn_State'){
        # data from Penn State experiments are nested with multiple nights per participant
        # also obtain mean, SD, 95% CI derived from mixed-effects models
        confusion_table <- data.frame('True.Wake' = array(NA, 4), 'True.Sleep' = array(NA, 4),
                                      row.names = c('Spectrum.Predicted.Wake', 'Spectrum.Predicted.Sleep', 'Classifier.Predicted.Wake', 'Classifier.Predicted.Sleep'))
        
        confusion_table['Spectrum.Predicted.Wake', 'True.Wake'] <- summary_string_mixed_effects(data$spectrum_TN * 100, data$participant_id, compute_ci = should_compute_ci)
        confusion_table['Spectrum.Predicted.Wake', 'True.Sleep'] <- summary_string_mixed_effects(data$spectrum_FN * 100, data$participant_id, compute_ci = should_compute_ci)
        confusion_table['Spectrum.Predicted.Sleep', 'True.Sleep'] <- summary_string_mixed_effects(data$spectrum_TP * 100, data$participant_id, compute_ci = should_compute_ci)
        confusion_table['Spectrum.Predicted.Sleep', 'True.Wake'] <- summary_string_mixed_effects(data$spectrum_FP * 100, data$participant_id, compute_ci = should_compute_ci)
        
        confusion_table['Classifier.Predicted.Wake', 'True.Wake'] <- summary_string_mixed_effects(data$classifier_TN * 100, data$participant_id, compute_ci = should_compute_ci)
        confusion_table['Classifier.Predicted.Wake', 'True.Sleep'] <- summary_string_mixed_effects(data$classifier_FN * 100, data$participant_id, compute_ci = should_compute_ci)
        confusion_table['Classifier.Predicted.Sleep', 'True.Sleep'] <- summary_string_mixed_effects(data$classifier_TP * 100, data$participant_id, compute_ci = should_compute_ci)
        confusion_table['Classifier.Predicted.Sleep', 'True.Wake'] <- summary_string_mixed_effects(data$classifier_FP * 100, data$participant_id, compute_ci = should_compute_ci)
        
        write.csv(confusion_table, sprintf('%s/%s_%s_%s_Confusion_Table_Mixed_Effects.csv', output_dir_path, model_name, label, interval))
  
      }
      
    }
  }
}

