import pandas as pd
import numpy as np
import preprocessing
import pathlib

# compute epoch counts to add to kept data stats table


def mean_sd_text(values):
    return f'{np.mean(values):.1f} ({np.std(values, ddof=1):.1f})'


data_folds = pd.read_csv('data_folds.csv')
data_folds = data_folds.loc[~data_folds['reject_recording']]  # exclude rejected recordings
data_folds = data_folds.reset_index(drop=True)

dataset_path = pathlib.Path('aligned_data/')

for idx in data_folds.index:
    csv_filename = data_folds.loc[idx, 'csv_filename']

    data = pd.read_csv(dataset_path / csv_filename)

    # two DeepSleeping datasets have mismarked 'lights off' times. This doesn't affect model training, but load
    # the correct 'lights off' times here
    participant_id = data_folds.loc[idx, 'participant_id']
    participant_session = data_folds.loc[idx, 'participant_session']
    data_set = data_folds.loc[idx, 'data_set']

    if np.isin(participant_id, ['1808DS', '1811DS']) & (participant_session == '3'):
        correct_lights = pd.read_csv(f'additional_staging/{participant_id}_Night_{participant_session}_Staging.csv')
        lights_off_rows = correct_lights['lights_off'] == 1
        lights_off_times = correct_lights['epoch_start_unix'][lights_off_rows]
        data.loc[np.isin(data['epoch_start_time'], lights_off_times), 'lights_off'] = 1

    if data_set == 'Ecosleep':
        sleep_wake = data['sleep_wake']
        primary_interval = preprocessing.identify_sleep_interval(sleep_wake, 180)
        data['lights_off'] = np.zeros(data.shape[0])
        data.loc[np.arange(primary_interval[0], primary_interval[1]+1), 'lights_off'] = 1

    valid_rows = (data['within_staged_period'] == 1) & \
                 ~np.isnan(data['actigraphy_activity']) & \
                 ~np.isnan(data['sleep_wake']) & \
                 (data['actigraphy_offwrist'] == 0) & \
                 ~np.isnan(data['actigraphy_device_classification'])

    sum_valid = np.sum(valid_rows)
    sum_valid_lights_off = np.sum(valid_rows & (data['lights_off'] == 1))

    data_folds.loc[idx, 'sum_valid'] = sum_valid
    data_folds.loc[idx, 'sum_valid_lights_off'] = sum_valid_lights_off


metadata_df_kept_grouped = data_folds.groupby('data_set')
mean_sd_valid_epochs = metadata_df_kept_grouped['sum_valid'].apply(mean_sd_text)
mean_sd_valid_lights_off_epochs = metadata_df_kept_grouped['sum_valid_lights_off'].apply(mean_sd_text)

epoch_count_table = pd.concat([mean_sd_valid_epochs, mean_sd_valid_lights_off_epochs], axis=1)

print(epoch_count_table)

dataset_order = ['SoundSleeping', 'DeepSleeping', 'SleepRestriction', 'MESA-COMMERCIAL-USE', 'Ecosleep']

dataset_display_names = {'SoundSleeping': 'Sound Sleeping',
                         'DeepSleeping': 'Deep Sleeping',
                         'SleepRestriction': 'Sleep Restriction',
                         'MESA-COMMERCIAL-USE': 'MESA',
                         'Ecosleep': 'Ecosleep'}

epoch_count_table = epoch_count_table.reindex(dataset_order)
epoch_count_table = epoch_count_table.rename(index=dataset_display_names)
epoch_count_table = epoch_count_table.sort_values('data_set')

epoch_count_table.to_csv('tables/Table_3_Addition_Epoch_Counts.csv', float_format='%.1f')


