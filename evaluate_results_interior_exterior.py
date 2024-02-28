import pandas as pd
import sklearn.metrics
import numpy as np
import sleepstats
import preprocessing
from pathlib import Path
import os


def roc_auc_if_avail(y_true, y_pred):
    num_true_classes = len(np.unique(y_true))
    if num_true_classes <= 1:
        return np.nan
    else:
        return sklearn.metrics.roc_auc_score(y_true, y_pred)


def roc_by_threshold_sklearn(y_true, y_pred, thresholds):
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    for t_idx, thresh in enumerate(thresholds):
        tpr[t_idx] = sklearn.metrics.recall_score(y_true, (y_pred >= thresh), pos_label=1)
        fpr[t_idx] = 1 - sklearn.metrics.recall_score(y_true, (y_pred >= thresh), pos_label=0)
    return fpr, tpr


# compute tpr and fpr manually rather than via SKLearn to be a bit faster for AUC Curve computation
def roc_by_threshold(y_true, y_prob, thresholds):
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    for t_idx, thresh in enumerate(thresholds):
        y_pred = (y_prob >= thresh)
        true_positive = (y_true == 1)
        tpr[t_idx] = np.mean(y_true[true_positive] == y_pred[true_positive])
        fpr[t_idx] = np.mean(y_true[~true_positive] != y_pred[~true_positive])
    return fpr, tpr


roc_curve_thresholds = np.arange(0, 1.001, .001)

# summarize both Penn State and MESA datasets, with both models
to_output = [{'label': 'Penn_State',
              'model_prefix': '24_Hour_09_10_2023T18_38_25',
              'datasets': ['DeepSleeping', 'Ecosleep', 'SleepRestriction', 'SoundSleeping']},
             {'label': 'MESA',
              'model_prefix': '24_Hour_09_10_2023T18_38_25',
              'datasets': ['MESA-COMMERCIAL-USE']},
             {'label': 'Penn_State',
              'model_prefix': '24_Hour_Bidirectional_09_09_2023T23_37_28',
              'datasets': ['DeepSleeping', 'Ecosleep', 'SleepRestriction', 'SoundSleeping']},
             {'label': 'MESA',
              'model_prefix': '24_Hour_Bidirectional_09_09_2023T23_37_28',
              'datasets': ['MESA-COMMERCIAL-USE']}]

# assemble output columns
epoch_output_types = ['classifier', 'spectrum', 'sadeh', 'scripps']
epoch_output_metrics = ['auc', 'accuracy', 'balanced_accuracy', 'sensitivity',
                        'specificity', 'ppv', 'npv', 'f1', 'mcc', 'pabak',
                        'TN', 'FN', 'TP', 'FP']
epoch_output_columns = [t + '_' + m for t in epoch_output_types for m in epoch_output_metrics]

night_output_types = ['true', 'classifier', 'spectrum', 'sadeh', 'scripps']
night_output_metrics = ['sol', 'waso', 'tst', 'se']
night_output_columns = [t + '_' + m for t in night_output_types for m in night_output_metrics]

# map the output name to the appropraite column in the input dataset
output_column_map = {'true': 'sleep_wake',
                  'classifier': 'classification_sleep',
                  'spectrum': 'actigraphy_device_classification',
                  'sadeh': 'sadeh_classification',
                  'scripps': 'scripps_classification'}


for data_dict in to_output:
    label = data_dict['label']
    datasets = data_dict['datasets']
    model_prefix = data_dict['model_prefix']

    output_path = Path(f'prediction_summary/{model_prefix}')
    output_path.mkdir(parents=True, exist_ok=True)

    data_folds = pd.read_csv('data_folds.csv')
    data_folds = data_folds.loc[np.isin(data_folds['data_set'], datasets), :]
    data_folds = data_folds.loc[~data_folds['reject_recording']]  # exclude rejected recordings
    data_folds = data_folds.reset_index(drop=True)

    dataset_path = f'output_with_additional_classifiers/{model_prefix}/predictions/'

    output_df_blank = data_folds.reindex(columns=data_folds.columns.tolist() + epoch_output_columns + night_output_columns)

    intervals = ['24_hour', 'lights_off']

    for k in intervals:

        auc_curve_output_dir = Path(f'{output_path}/roc_curve_data/{label}_{k}/')
        auc_curve_output_dir.mkdir(parents=True, exist_ok=True)

        output_df = output_df_blank.copy()

        for idx, d in enumerate(output_df['csv_filename']):

            if (idx % 50) == 0:
                print(f'On record {idx}')

            filename = model_prefix + '_Predictions_' + os.path.splitext(d)[0] + '_additional_classifiers.csv'
            data = pd.read_csv(dataset_path + filename)
            # boolean to int for consistency with other output types
            data['classification_sleep'] = data['classification_sleep'].astype('int')

            # two DeepSleeping datasets have mismarked 'lights off' times, load the correct 'lights off' times here
            # lights off indicators were not used for model training, so only effects evaluation
            participant_id = output_df.loc[idx, 'participant_id']
            participant_session = output_df.loc[idx, 'participant_session']
            data_set = output_df.loc[idx, 'data_set']

            if np.isin(participant_id, ['1808DS', '1811DS']) & (participant_session == '3'):
                correct_lights = pd.read_csv(f'additional_staging/{participant_id}_Night_{participant_session}_Staging.csv')
                lights_off_rows = correct_lights['lights_off'] == 1
                lights_off_times = correct_lights['epoch_start_unix'][lights_off_rows]
                data.loc[np.isin(data['epoch_start_time'], lights_off_times), 'lights_off'] = 1

            # Lights-Off was not observed for Ecosleep, estimate it from the PSG sleep/wake values
            if data_set == 'Ecosleep':
                sleep_wake = data['sleep_wake']
                primary_interval = preprocessing.identify_sleep_interval(sleep_wake, 180)
                data['lights_off'] = np.zeros(data.shape[0])
                data.loc[np.arange(primary_interval[0], primary_interval[1]+1), 'lights_off'] = 1

            valid_rows = (data['within_staged_period'] == 1) & \
                ~np.isnan(data['classification_sleep']) & \
                ~np.isnan(data['actigraphy_activity']) & \
                ~np.isnan(data['sleep_wake']) & \
                (data['actigraphy_offwrist'] == 0) & \
                ~np.isnan(data['actigraphy_device_classification']) & \
                ~np.isnan(data['sadeh_classification']) & \
                ~np.isnan(data['scripps_classification'])

            if k == 'lights_off':
                valid_rows = valid_rows & (data['lights_off'] == 1)

            if np.any(valid_rows):
                y_true = data.loc[valid_rows, 'sleep_wake']
                y_prob = data.loc[valid_rows, 'probability_sleep']

                # only the TCN Classifier outputs probabilities which can be used to calculate AUC
                output_df.loc[idx, 'classifier_auc'] = roc_auc_if_avail(y_true, y_prob)

                # compute AUC curve at common intervals
                fpr, tpr = roc_by_threshold(y_true, y_prob, roc_curve_thresholds)
                auc_file_path = f'{auc_curve_output_dir}/{label}_{k}_{participant_id}_{participant_session}_auc_values.csv'
                auc_df = pd.DataFrame({'participant_id': participant_id,
                                       'participant_session': participant_session,
                                       'data_set': data_set,
                                       'fpr': fpr,
                                       'tpr': tpr,
                                       'thresholds': roc_curve_thresholds})
                auc_df.to_csv(auc_file_path, index=False)

                classifications = data.loc[valid_rows, 'classification_sleep']
                device_class = data.loc[valid_rows, 'actigraphy_device_classification']

                for t in epoch_output_types:
                    classification_column_name = output_column_map[t]
                    classifications = data.loc[valid_rows, classification_column_name]

                    # classifier epoch metrics
                    output_df.loc[idx, t + '_accuracy'] = sklearn.metrics.accuracy_score(y_true, classifications)
                    output_df.loc[idx, t + '_balanced_accuracy'] = sklearn.metrics.balanced_accuracy_score(y_true,classifications)
                    output_df.loc[idx, t + '_sensitivity'] = sklearn.metrics.recall_score(y_true, classifications,pos_label=1)
                    output_df.loc[idx, t + '_specificity'] = sklearn.metrics.recall_score(y_true, classifications,pos_label=0)
                    output_df.loc[idx, t + '_ppv'] = sklearn.metrics.precision_score(y_true, classifications, pos_label=1)
                    output_df.loc[idx, t + '_npv'] = sklearn.metrics.precision_score(y_true, classifications, pos_label=0)
                    output_df.loc[idx, t + '_f1'] = sklearn.metrics.f1_score(y_true, classifications, pos_label=1)
                    output_df.loc[idx, t + '_mcc'] = sklearn.metrics.matthews_corrcoef(y_true, classifications)
                    output_df.loc[idx, t + '_pabak'] = (2 * sklearn.metrics.accuracy_score(y_true, classifications)) - 1

                    output_confusion_matrix = sklearn.metrics.confusion_matrix(y_true, classifications, normalize='all')
                    output_df.loc[idx, t + '_TN'] = output_confusion_matrix[0, 0]
                    output_df.loc[idx, t + '_FN'] = output_confusion_matrix[1, 0]
                    output_df.loc[idx, t + '_TP'] = output_confusion_matrix[1, 1]
                    output_df.loc[idx, t + '_FP'] = output_confusion_matrix[0, 1]

                for t in night_output_types:
                    classification_column_name = output_column_map[t]
                    classifications = data.loc[valid_rows, classification_column_name]

                    output_df.loc[idx, t + '_waso'] = sleepstats.waso(classifications == 1, 30)
                    output_df.loc[idx, t + '_tst'] = sleepstats.total_sleep_time(classifications == 1, 30)

                    if output_df.loc[idx, 'data_set'] == 'Ecosleep':
                        # if Ecosleep, mask values that can't be computed without a true 'lights off' period indicated
                        output_df.loc[idx, t + '_sol'] = np.nan
                        output_df.loc[idx, t + '_se'] = np.nan
                    else:
                        output_df.loc[idx, t + '_sol'] = sleepstats.sleep_onset_latency(classifications == 1, 30)
                        output_df.loc[idx, t + '_se'] = sleepstats.sleep_efficiency(classifications == 1)

        output_df.to_csv(f'{output_path}/{model_prefix}_{label}_{k}_by_record.csv')

        condensed_output = output_df.loc[:, epoch_output_columns]
        summary_df = condensed_output.apply(lambda x: f'{np.mean(x):.3f} ({np.std(x):.3f})')
        summary_df.to_csv(f'{output_path}/{model_prefix}_{label}_{k}_summary.csv')
