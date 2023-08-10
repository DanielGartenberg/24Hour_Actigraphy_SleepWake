import pathlib
import json
import pandas as pd
import numpy as np
import os
import preprocessing

# fraction of unscored or missing epochs that results in removal of the recording from further analyses
PSG_REJECTION_THRESHOLD = .25
ACTIGRAPHY_REJECTION_THRESHOLD = .25

# how many contiguous epochs of 0 activity counts are required to consider the period 'off-wrist'
OFF_WRIST_DETECTION_MINUTES = 120
EPOCH_LENGTH_SECONDS = 30
off_wrist_epochs = OFF_WRIST_DETECTION_MINUTES * (60 / EPOCH_LENGTH_SECONDS)

PRESTAGING_ACTIGRAPHY_HOURS = 4
POSTSTAGING_ACTIGRAPHY_HOURS = 4

wake_labels = ['Wake', 'ObservedWake']
sleep_labels = ['N1', 'N2', 'N3', 'N4', 'REM']
unscored_labels = ['Unscored']

dataset_path = 'data/'
datasets = pathlib.Path(dataset_path).rglob('*.json')

aligned_dataset_path = 'aligned_data/' # path to export aligned datafiles

metadata = list()

for idx, d in enumerate(datasets):

    if (idx % 10) == 0:
        print(f'Processing dataset {idx}')  # display progress

    with open(d) as f:

        data = json.load(f)
        meta_values = data['metadata']['participant'].copy()

        # add pre-staging timestamps, to associate actigraphy timestamps within the prestaging period
        staging_start_time = data['sleep_staging']['epoch_start_time'][0]
        pre_staging_start = staging_start_time - (PRESTAGING_ACTIGRAPHY_HOURS * 3600)
        pre_staging_timestamps = np.arange(pre_staging_start, staging_start_time, EPOCH_LENGTH_SECONDS)
        num_pre_staging_epochs = len(pre_staging_timestamps)
        num_staged_epochs = len(data['sleep_staging']['epoch_start_time'])

        # append the post-staging data
        data['sleep_staging']['epoch_start_time'] = pre_staging_timestamps.tolist() + data['sleep_staging']['epoch_start_time']
        data['sleep_staging']['epoch_stage_label'] = (['Unscored'] * num_pre_staging_epochs) + data['sleep_staging']['epoch_stage_label']
        data['sleep_staging']['lights_off'] = ([None] * num_pre_staging_epochs) + data['sleep_staging']['lights_off']
        data['sleep_staging']['within_staged_period'] = ([0] * num_pre_staging_epochs) + ([1] * num_staged_epochs)
        data['sleep_staging']['within_pre_staging'] = ([1] * num_pre_staging_epochs) + ([0] * num_staged_epochs)
        data['sleep_staging']['within_post_staging'] = [0] * (num_pre_staging_epochs + num_staged_epochs)

        # add post-staging timestamps
        staging_end_time = data['sleep_staging']['epoch_start_time'][-1]
        post_staging_end = staging_end_time + (POSTSTAGING_ACTIGRAPHY_HOURS * 3600)
        post_staging_timestamps = np.arange(staging_end_time + EPOCH_LENGTH_SECONDS, post_staging_end + EPOCH_LENGTH_SECONDS, EPOCH_LENGTH_SECONDS)
        num_post_staging_epochs = len(post_staging_timestamps)

        # append the post-staging data
        data['sleep_staging']['epoch_start_time'] = data['sleep_staging']['epoch_start_time'] + post_staging_timestamps.tolist()
        data['sleep_staging']['epoch_stage_label'] = data['sleep_staging']['epoch_stage_label'] + (['Unscored'] * num_post_staging_epochs)
        data['sleep_staging']['lights_off'] = data['sleep_staging']['lights_off'] + ([None] * num_post_staging_epochs)
        data['sleep_staging']['within_staged_period'] = data['sleep_staging']['within_staged_period'] + ([0] * num_post_staging_epochs)
        data['sleep_staging']['within_pre_staging'] = data['sleep_staging']['within_pre_staging'] + ([0] * num_post_staging_epochs)
        data['sleep_staging']['within_post_staging'] = data['sleep_staging']['within_post_staging'] + ([1] * num_post_staging_epochs)

        staging_times = data['sleep_staging']['epoch_start_time']
        staging_start_time = staging_times[0]
        staging_end_time = staging_times[-1]

        actigraphy_times = np.array(data['actigraphy']['unix_time'])
        actigraphy_activity = np.array(data['actigraphy']['activity']).astype('float32')
        actigraphy_offwrist = np.array(data['actigraphy']['offwrist']).astype('float32')

        # classifications produced by the Spectrum device
        actigraphy_classifier = np.array(data['actigraphy']['wake']).astype('float32')
        # the spectrum classifier output natively uses 0 for sleep, and 1 for wake.
        # Swap the values here to be consistent with other measures where sleep is considered the 'target'
        sleep_idx = np.where(actigraphy_classifier == 0)
        wake_idx = np.where(actigraphy_classifier == 1)
        actigraphy_classifier[sleep_idx] = 1
        actigraphy_classifier[wake_idx] = 0

        # some actigraphy files may contain a large amount of data (ie 1 week or more).
        # reduce actigraphy to only relevant data with +- .5 hour buffer relative to staging, to reduce processing time
        buffer_seconds = 1800
        actigraphy_idx_to_retain = (actigraphy_times > (staging_start_time - buffer_seconds)) \
                                   & (actigraphy_times < (staging_end_time + buffer_seconds))
        actigraphy_times = actigraphy_times[actigraphy_idx_to_retain]
        actigraphy_activity = actigraphy_activity[actigraphy_idx_to_retain]
        actigraphy_offwrist = actigraphy_offwrist[actigraphy_idx_to_retain]
        actigraphy_classifier = actigraphy_classifier[actigraphy_idx_to_retain]

        # use contiguous periods of no activity (activity counts of 0) to additionally identify periods of off-wrist
        actigraphy_automated_offwrist = np.zeros(actigraphy_offwrist.shape)
        auto_off_wrist_indices = preprocessing.identify_off_wrist(actigraphy_activity, off_wrist_epochs)
        if np.any(auto_off_wrist_indices):
            actigraphy_automated_offwrist[auto_off_wrist_indices] = 1

        # identify the lag between staging and actigraphy that results in the optimal correspondance
        # preprocess the values by converting the sleep staging to a binary indicator of wake
        sleep_staging = np.array(data['sleep_staging']['epoch_stage_label'])
        sleep_wake_binary = np.full(len(sleep_staging), np.nan)
        sleep_wake_binary[np.isin(sleep_staging, wake_labels)] = 0.
        sleep_wake_binary[np.isin(sleep_staging, sleep_labels)] = 1.
        sleep_wake_binary[np.isin(sleep_staging, unscored_labels)] = np.nan

        meta_values['unique_classes'] = np.unique(sleep_wake_binary[~np.isnan(sleep_wake_binary)])

        # log the combination of activity NaN and device offwrist prior to the use of the off-wrist algorithm,
        # to determine the unique contribution of automated off-wrist
        pre_algorithm_off_wrist_or_nan = np.isnan(actigraphy_activity) | actigraphy_offwrist.astype('bool')

        # mask any 'offwrist' actigraphy activity values with nan
        actigraphy_merged_off_wrist = actigraphy_offwrist.astype('bool') | actigraphy_automated_offwrist.astype('bool')
        actigraphy_activity[actigraphy_merged_off_wrist] = np.nan

        # assemble 2D arrays containing both timestamps and data
        sleep_wake_data = np.vstack([data['sleep_staging']['epoch_start_time'], sleep_wake_binary]).T
        actigraphy_data = np.vstack([actigraphy_times, actigraphy_activity]).T

        # the lag value is the position of actigraphy data relative to sleep wake data
        # subtract this value from all actigraphy timestamps to correct the lag.
        try:
            meta_values['lag'] = preprocessing.identify_lag_threshold(sleep_wake_data, actigraphy_data,
                                                                      max_lag=20, max_sequence_length=20)
        except Exception as e:
            print(f'Skipped {d}')
            meta_values['exception'] = e
            meta_values['lag'] = np.nan

        # apply the identified lag, or apply no correction if a lag value couldn't be calculated
        lag_value = meta_values['lag']
        if np.isnan(lag_value):
            lag_value = 0
        actigraphy_times_lag_corrected = actigraphy_times - lag_value

        # for each sleep epoch, find closest actigraphy epoch that is within half of an epoch in distance
        # for each sleep epoch, find closest actigraphy epoch
        actigraphy_staging_idx = preprocessing.associate_timestamps(staging_times, actigraphy_times_lag_corrected)

        actigraphy_activity_aligned = actigraphy_activity[actigraphy_staging_idx]
        actigraphy_classifier_aligned = actigraphy_classifier[actigraphy_staging_idx]
        actigraphy_offwrist_aligned = actigraphy_offwrist[actigraphy_staging_idx]
        actigraphy_automated_offwrist_aligned = actigraphy_automated_offwrist[actigraphy_staging_idx]
        actigraphy_merged_off_wrist_aligned = actigraphy_merged_off_wrist[actigraphy_staging_idx]

        # log the proportion of epochs uniquely identified as off-wrist by the algorithm
        pre_algorithm_off_wrist_or_nan_aligned = pre_algorithm_off_wrist_or_nan[actigraphy_staging_idx]
        unique_algorithm_id_offwrist = (actigraphy_automated_offwrist_aligned.astype('bool') &
                                        ~pre_algorithm_off_wrist_or_nan_aligned)

        # masked values are any actigraphy epochs more than 15 seconds from corresponding sleep epoch
        if np.any(actigraphy_staging_idx.mask):
            actigraphy_activity_aligned[actigraphy_staging_idx.mask] = np.nan
            actigraphy_classifier_aligned[actigraphy_staging_idx.mask] = np.nan
            actigraphy_offwrist_aligned[actigraphy_staging_idx.mask] = np.nan
            actigraphy_automated_offwrist_aligned[actigraphy_staging_idx.mask] = np.nan
            actigraphy_merged_off_wrist_aligned[actigraphy_staging_idx.mask] = np.nan


        # determine the amount of data within the region staged by RPSGT that was missing or unscored
        sleep_staging = np.array(data['sleep_staging']['epoch_stage_label'])
        staged_epochs = np.array(data['sleep_staging']['within_staged_period']) == 1
        pre_staging_epochs = np.array(data['sleep_staging']['within_pre_staging']) == 1
        post_staging_epochs = np.array(data['sleep_staging']['within_post_staging']) == 1
        observed_data = sleep_staging[(sleep_staging != 'ObservedWake') & staged_epochs]
        meta_values['fraction_unscored_observed_psg'] = np.mean(observed_data == 'Unscored')
        meta_values['fraction_unscored_all_psg'] = np.mean(sleep_staging[staged_epochs] == 'Unscored')

        # all off-wrist values were also set to NaN activity counts in a previous step
        meta_values['fraction_nan_actigraphy_staged'] = np.mean(np.isnan(actigraphy_activity_aligned[staged_epochs]))
        meta_values['fraction_off_wrist_spectrum_staged'] = np.nanmean(actigraphy_offwrist_aligned[staged_epochs])
        meta_values['fraction_off_wrist_algorithm_staged'] = np.nanmean(actigraphy_automated_offwrist_aligned[staged_epochs])
        meta_values['fraction_off_wrist_merged_staged'] = np.nanmean(actigraphy_merged_off_wrist_aligned[staged_epochs])
        meta_values['fraction_off_wrist_algorithm_unique_staged'] = np.nanmean(unique_algorithm_id_offwrist[staged_epochs])

        meta_values['fraction_nan_actigraphy_pre_staging'] = np.mean(np.isnan(actigraphy_activity_aligned[pre_staging_epochs]))
        meta_values['fraction_off_wrist_spectrum_pre_staging'] = np.nanmean(actigraphy_offwrist_aligned[pre_staging_epochs])
        meta_values['fraction_off_wrist_algorithm_pre_staging'] = np.nanmean(actigraphy_automated_offwrist_aligned[pre_staging_epochs])
        meta_values['fraction_off_wrist_merged_pre_staging'] = np.nanmean(actigraphy_merged_off_wrist_aligned[pre_staging_epochs])
        meta_values['fraction_off_wrist_algorithm_unique_pre_staging'] = np.nanmean(unique_algorithm_id_offwrist[pre_staging_epochs])

        meta_values['fraction_nan_actigraphy_post_staging'] = np.mean(np.isnan(actigraphy_activity_aligned[post_staging_epochs]))
        meta_values['fraction_off_wrist_spectrum_post_staging'] = np.nanmean(actigraphy_offwrist_aligned[post_staging_epochs])
        meta_values['fraction_off_wrist_algorithm_post_staging'] = np.nanmean(actigraphy_automated_offwrist_aligned[post_staging_epochs])
        meta_values['fraction_off_wrist_merged_post_staging'] = np.nanmean(actigraphy_merged_off_wrist_aligned[post_staging_epochs])
        meta_values['fraction_off_wrist_algorithm_unique_post_staging'] = np.nanmean(unique_algorithm_id_offwrist[post_staging_epochs])

        meta_values['fraction_nan_actigraphy'] = np.mean(np.isnan(actigraphy_activity_aligned))
        meta_values['fraction_off_wrist_spectrum'] = np.nanmean(actigraphy_offwrist_aligned)
        meta_values['fraction_off_wrist_algorithm'] = np.nanmean(actigraphy_automated_offwrist_aligned)
        meta_values['fraction_off_wrist_merged'] = np.nanmean(actigraphy_merged_off_wrist_aligned)
        meta_values['fraction_off_wrist_algorithm_unique'] = np.nanmean(unique_algorithm_id_offwrist)

        # output a new aligned data file
        epoch_data = data['sleep_staging'].copy()
        epoch_data['sleep_wake'] = sleep_wake_binary
        epoch_data['actigraphy_activity'] = actigraphy_activity_aligned
        epoch_data['actigraphy_device_classification'] = actigraphy_classifier_aligned
        epoch_data['actigraphy_offwrist'] = actigraphy_merged_off_wrist_aligned.astype('int') # from logical

        aligned_data = pd.DataFrame(epoch_data)
        csv_filename = d.stem + '.csv'
        aligned_data.to_csv(aligned_dataset_path + d.stem + '.csv', index=None)

        # log the metadata
        meta_values['csv_filename'] = csv_filename
        meta_values['json_filename'] = os.path.basename(d)
        metadata.append(meta_values)


# Two SleepRestriction datasets were not staged due to the poor quality of data. Include them manually here.
metadata.append({'data_set': 'SleepRestriction',
                 'data_source': 'PennState',
                 'participant_id': '1604SR',
                 'participant_session': 'SR01',
                 'fraction_unscored_observed_psg': 1,
                 'fraction_unscored_all_psg': 1})

metadata.append({'data_set': 'SleepRestriction',
                 'data_source': 'PennState',
                 'participant_id': '1711SR',
                 'participant_session': 'SR03',
                 'fraction_unscored_observed_psg': 1,
                 'fraction_unscored_all_psg': 1})

metadata_df = pd.DataFrame(metadata)

metadata_df['reject_from_staging'] = metadata_df.fraction_unscored_observed_psg >= PSG_REJECTION_THRESHOLD
metadata_df['reject_from_actigraphy'] = metadata_df.fraction_nan_actigraphy_staged >= ACTIGRAPHY_REJECTION_THRESHOLD
metadata_df['reject_recording'] = (metadata_df.reject_from_staging | metadata_df.reject_from_actigraphy)

metadata_df.to_csv('rejection_metadata.csv', index=None)
