import numpy as np
import pandas as pd
import random

rng = np.random.default_rng()
num_folds = 5

metadata_df = pd.read_csv('rejection_metadata.csv')
metadata_df.loc[metadata_df.reject_recording, 'fold'] = np.nan  # rejected recordings will not be part of any fold

# generate folds such that:
# Each fold has an approximately equal distribution of the sub-datsets
# As some participants have several recordings, always place all of a given participants data in a single fold
# Log the folds to a file for reproducability in-case models tested on a given fold are later combined / stacked

for d in np.unique(metadata_df.data_set):
    dataset_rows_not_rejected = (metadata_df['data_set'] == d) & ~metadata_df['reject_recording']
    dataset_participants = np.unique(metadata_df.loc[dataset_rows_not_rejected, 'participant_id'])
    # permute so that participants adjacent in time are not necessarily in the same fold
    data_set_participants_permuted = rng.permutation(dataset_participants)
    fold_splits = np.array_split(data_set_participants_permuted, num_folds)
    # as the final splits may have fewer participants than prior, shuffle order of splits in place
    random.shuffle(fold_splits)
    for fold_idx, fold_participants in enumerate(fold_splits):
        # label recordings from this dataset, who are (1) not rejected (2) one of the participants in the fold
        rows_to_label = dataset_rows_not_rejected & np.isin(metadata_df['participant_id'], fold_participants)
        metadata_df.loc[rows_to_label, 'fold'] = fold_idx

        # write out the metadata file with partitions labeled
        metadata_df.to_csv('data_folds.csv', index=False)
