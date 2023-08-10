import numpy as np
import pandas as pd

metadata_df = pd.read_csv('rejection_metadata.csv')


def mean_sd_text(values):
    return f'{np.mean(values):.1f} ({np.std(values, ddof=1):.1f})'


# counts of number of participants prior to rejection
metadata_df_unique_pre_rejection = metadata_df.drop_duplicates(subset=['participant_id', 'data_set'])
participants_per_dataset_pre_rejection = metadata_df_unique_pre_rejection.groupby('data_set').size()
participants_per_dataset_pre_rejection.name = 'participants prior to rejection'

# counts of number of nights prior to rejection
grouped_df = metadata_df.groupby('data_set')
nights_per_dataset_pre_rejection = grouped_df.size()
nights_per_dataset_pre_rejection.name = 'nights prior to rejection'

# compute demographics and summary statistics for retained nights
metadata_kept_df = metadata_df.loc[~metadata_df['reject_recording']].copy()

# some participants have more than one night,
# eliminate duplicates prior to computing demographics so each participant is only counted once
metadata_df_unique = metadata_kept_df.drop_duplicates(subset=['participant_id', 'data_set'])
mean_sd_age = metadata_df_unique.groupby('data_set')['age'].apply(mean_sd_text)
participants_per_dataset_post_rejection = metadata_df_unique.groupby('data_set').size()
participants_per_dataset_post_rejection.name = 'participants per dataset post rejection'
gender_totals = metadata_df_unique.groupby('data_set')['gender'].apply(lambda x: (np.sum(x == 'M'), np.sum(x == 'F')))

# total nights per dataset
nights_per_dataset = metadata_kept_df.groupby('data_set').size()
nights_per_dataset.name = 'nights per dataset'

# amount of data rejected
rejected_due_to_staging = grouped_df['reject_from_staging'].apply(lambda x: np.sum(x))
rejected_due_to_actigraphy = grouped_df['reject_from_actigraphy'].apply(lambda x: np.sum(x))
rejected_due_to_either = grouped_df['reject_recording'].apply(lambda x: np.sum(x))

# stats on datasets retained
metadata_kept_df.loc[:, metadata_kept_df.columns.str.startswith('fraction')] *= 100 # convert fractions to percent
metadata_df_kept_grouped = metadata_kept_df.groupby('data_set')
mean_sd_unscored_all = metadata_df_kept_grouped['fraction_unscored_all_psg'].apply(mean_sd_text)
mean_sd_unscored_observed = metadata_df_kept_grouped['fraction_unscored_observed_psg'].apply(mean_sd_text)
mean_sd_missing_actigraphy = metadata_df_kept_grouped['fraction_nan_actigraphy_staged'].apply(mean_sd_text)
mean_sd_missing_actigraphy_pre_staging = metadata_df_kept_grouped['fraction_nan_actigraphy_pre_staging'].apply(mean_sd_text)
mean_sd_missing_actigraphy_post_staging = metadata_df_kept_grouped['fraction_nan_actigraphy_post_staging'].apply(mean_sd_text)

mean_sd_offwrist_spectrum = metadata_df_kept_grouped['fraction_off_wrist_spectrum'].apply(mean_sd_text)
mean_sd_offwrist_merged = metadata_df_kept_grouped['fraction_off_wrist_merged'].apply(mean_sd_text)

nights_per_dataset_post_rejection = metadata_df_kept_grouped.size()
nights_per_dataset_post_rejection.name = 'nights following rejection'

# lag stats
mean_no_lag = metadata_df_kept_grouped['lag'].apply(lambda x: np.mean(np.isnan(x))) * 100
mean_no_lag.name = 'percent that could not have lag algorithm applied'

mean_sd_lag = metadata_df_kept_grouped['lag'].apply(lambda x: f'{np.nanmean(x):.1f} ({np.nanstd(x, ddof=1):.1f})')
mean_sd_lag.name = 'average lag identified (seconds)'

mean_zero_lag = metadata_df_kept_grouped['lag'].apply(lambda x: np.nanmean(x == 0)) * 100
mean_zero_lag.name = 'percent of records where lag identified was 0'

# demographic table
dataset_order = ['SoundSleeping', 'DeepSleeping', 'SleepRestriction', 'MESA-COMMERCIAL-USE', 'Ecosleep']
dataset_display_names = {'SoundSleeping': 'Sound Sleeping',
                         'DeepSleeping': 'Deep Sleeping',
                         'SleepRestriction': 'Sleep Restriction',
                         'MESA-COMMERCIAL-USE': 'MESA',
                         'Ecosleep': 'Ecosleep'}


gender_totals_string = pd.Series([f'{m}M / {f}F' for m, f in gender_totals], index=gender_totals.index)
gender_totals_string.name = 'gender'

# statistics of rejected datasets
rejection_table = pd.concat([participants_per_dataset_pre_rejection,
                             nights_per_dataset_pre_rejection,
                             rejected_due_to_staging,
                             rejected_due_to_actigraphy,
                             rejected_due_to_either,
                             participants_per_dataset_post_rejection,
                             nights_per_dataset_post_rejection], axis=1)

rejection_table = rejection_table.reindex(dataset_order)
rejection_table = rejection_table.rename(index=dataset_display_names)
rejection_table = rejection_table.sort_values('data_set')
rejection_table.to_csv('tables/Table_2_Rejected_Data.csv')

# statistics on the datasets that remain following rejection
kept_stats_table = pd.concat([mean_sd_age,
                              gender_totals_string,
                              mean_sd_unscored_all,
                              mean_sd_missing_actigraphy,
                              mean_sd_missing_actigraphy_pre_staging,
                              mean_sd_missing_actigraphy_post_staging,
                              mean_no_lag,
                              mean_sd_lag,
                              mean_zero_lag],
                             axis=1)

kept_stats_table = kept_stats_table.reindex(dataset_order)
kept_stats_table = kept_stats_table.rename(index=dataset_display_names)
kept_stats_table = kept_stats_table.sort_values('data_set')
kept_stats_table.to_csv('tables/Table_3_Statistics_on_Retained_Data.csv', float_format='%.1f')
