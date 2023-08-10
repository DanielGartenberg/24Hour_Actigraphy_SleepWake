import numpy as np
from scipy import stats
import warnings


def associate_timestamps(sleep_staging_times, actigraphy_times, epoch_length_seconds=30):
    half_epoch = epoch_length_seconds / 2
    max_act_idx = len(actigraphy_times) - 1
    associated_timestamps_pre_limit = np.searchsorted((actigraphy_times + half_epoch), sleep_staging_times)
    associated_timestamps = np.minimum(associated_timestamps_pre_limit, max_act_idx)
    abs_timestamp_distance = np.abs(actigraphy_times[associated_timestamps] - sleep_staging_times)
    masked_timestamps = np.ma.masked_where(abs_timestamp_distance > half_epoch, associated_timestamps)
    return masked_timestamps


def associate_timestamps_previous(sleep_staging_times, actigraphy_times, epoch_length_seconds=30):
    half_epoch = epoch_length_seconds / 2
    shifted_indices = np.zeros(len(sleep_staging_times)).astype('int')
    shifted_distance = np.zeros(len(sleep_staging_times))
    for seq_idx, seq_time in enumerate(sleep_staging_times):
        actigraphy_offsets = np.abs(actigraphy_times - seq_time)
        shifted_indices[seq_idx] = np.argmin(actigraphy_offsets)
        shifted_distance[seq_idx] = actigraphy_offsets[shifted_indices[seq_idx]]
    masked_timestamps = np.ma.masked_where(shifted_distance > half_epoch, shifted_indices)
    return masked_timestamps



def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    # from StackOverflow:
    # https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array

    # Find the indices of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def locate_contiguous(sequence):
    """For a boolean array sequence, locate each sequence of True elements."""
    state_changes = np.diff(sequence.astype('int'), prepend=0, append=0)
    sequence_onsets = np.nonzero(state_changes == 1)
    sequence_offsets = np.nonzero(state_changes == -1)
    return np.vstack([sequence_onsets, sequence_offsets]).T


def identify_off_wrist(actigraphy_activity, contiguous_epoch_threshold):
    """For a boolean array sequence, locate each sequence of True elements."""
    contiguous_zeros = locate_contiguous(actigraphy_activity == 0)
    segment_lengths = np.squeeze(np.diff(contiguous_zeros, 1), 1)
    segments_over_threshold = contiguous_zeros[segment_lengths >= contiguous_epoch_threshold]
    if np.any(segments_over_threshold):
        sequences_over_threshold = [np.arange(start, end) for start, end in segments_over_threshold]
        return np.hstack(sequences_over_threshold)
    else:
        return np.array([])


def identify_lag_threshold_previous(sleep_wake, actigraphy, max_lag=10):
    """Identify lag between sleep-wake and actigraphy using a threshold."""
    activity_threshold = 15
    max_sequence_length = 3
    significance_threshold = .05
    epoch_length_seconds = 30

    wake_idx = 0
    sleep_idx = 1

    contains_sleep = np.any(sleep_wake[:, 1] == sleep_idx)
    contains_wake = np.any(sleep_wake[:, 1] == wake_idx)

    if not (contains_sleep and contains_wake):
        if contains_sleep:
            data_contains = 'sleep'
        elif contains_wake:
            data_contains = 'wake'
        warnings.warn(f'The sleep wake data contains only {data_contains}, returning NaN')
        return np.nan

    # trim all epochs before the first or after the last sleep epoch
    # this is because a leading or trailing run of wake in the sequence cannot be known to be a given length
    # i.e. a single wake epoch prior to the first sleep epoch could be been preceded by wake
    sleep_indices = np.where(sleep_wake[:, 1] == sleep_idx)
    sleep_wake = sleep_wake[np.arange(sleep_indices[0][0], sleep_indices[0][-1] + 1), :]

    missing_epochs = np.isnan(sleep_wake[:, 1])
    sleep_wake[missing_epochs, 1] = 0.

    wake_or_missing = (sleep_wake[:, 1] == 0) | missing_epochs
    contiguous_sequences = locate_contiguous(wake_or_missing)
    sequence_length = np.squeeze(np.diff(contiguous_sequences, 1), 1)
    sequences_are_valid_length = sequence_length <= max_sequence_length
    valid_length_sequences = contiguous_sequences[sequences_are_valid_length]
    if np.any(valid_length_sequences):
        # expand, combine, remove sequence positions that were originally missing
        sequences_expanded = [np.arange(x, y) for x, y in valid_length_sequences]
        sequences_combined = np.hstack(sequences_expanded)
        sequence_indices = sequences_combined[np.isin(sequences_combined, np.where(~missing_epochs))]
    else:
        sequence_indices = np.array([])

    # no valid indices if either there were no valid length sequences, or all valid length were composed of missing
    if len(sequence_indices) == 0:
        warnings.warn('There are no Wake sequences which match the criteria, returning NaN')
        return np.nan

    sequence_timestamps = sleep_wake[sequence_indices, 0]

    potential_lag_values = np.arange(-1 * max_lag, max_lag + 1) * epoch_length_seconds
    counts = np.zeros((2, len(potential_lag_values)))

    activity_timestamps = actigraphy[:, 0]
    activity_counts = actigraphy[:, 1]

    # shift the activity counts +- relative to staging references
    for idx, shift in enumerate(potential_lag_values):
        shifted_actigraphy_times = activity_timestamps - shift
        shifted_indices = np.zeros(len(sequence_timestamps)).astype('int')
        shifted_distance = np.zeros(len(sequence_timestamps))
        for seq_idx, seq_time in enumerate(sequence_timestamps):
            actigraphy_offsets = np.abs(shifted_actigraphy_times - seq_time)
            shifted_indices[seq_idx] = np.argmin(actigraphy_offsets)
            shifted_distance[seq_idx] = actigraphy_offsets[shifted_indices[seq_idx]]

        shifted_indices = shifted_indices[shifted_distance <= (epoch_length_seconds / 2)]  # check distance validity
        counts_at_shift = activity_counts[shifted_indices]
        num_above_threshold = np.sum(counts_at_shift >= activity_threshold)
        num_below_threshold = np.sum(counts_at_shift < activity_threshold)
        counts[:, idx] = (num_above_threshold, num_below_threshold)

    # in the event of no actigraphy epochs above threshold (for example due to missing data),
    # the denominator here may be zero.
    with np.errstate(invalid='ignore'):
        proportions = counts[0, :] / np.sum(counts, 0)

    # compare each shifted value to 0
    zero_shift_idx = np.where(potential_lag_values == 0)[0][0]
    zero_shift_proportion = proportions[zero_shift_idx]
    shift_greater_match = proportions > zero_shift_proportion

    p_values = np.ones(len(potential_lag_values))
    for s in range(len(potential_lag_values)):
        if shift_greater_match[s]:
            p_values[s] = stats.boschloo_exact(counts[:, [s, zero_shift_idx]]).pvalue

    # identify if any of the time shifts generated a significant increase in the
    significant_shift = p_values < significance_threshold
    potential_shift_idx = significant_shift & shift_greater_match
    if np.any(potential_shift_idx):
        subset_values = potential_lag_values[potential_shift_idx]
        subset_coefs = proportions[potential_shift_idx]
        lag_value = subset_values[np.argmax(subset_coefs)]
    else:
        lag_value = 0

    return lag_value


def identify_lag_threshold(sleep_wake, actigraphy, max_lag=10, max_sequence_length=3):
    """Identify lag between sleep-wake and actigraphy using a threshold."""
    activity_threshold = 15
    significance_threshold = .05
    epoch_length_seconds = 30

    wake_idx = 0
    sleep_idx = 1

    contains_sleep = np.any(sleep_wake[:, 1] == sleep_idx)
    contains_wake = np.any(sleep_wake[:, 1] == wake_idx)

    if not (contains_sleep and contains_wake):
        if contains_sleep:
            data_contains = 'sleep'
        elif contains_wake:
            data_contains = 'wake'
        warnings.warn(f'The sleep wake data contains only {data_contains}, returning NaN')
        return np.nan

    # trim all epochs before the first or after the last sleep epoch
    # this is because a leading or trailing run of wake in the sequence cannot be known to be a given length
    # i.e. a single wake epoch prior to the first sleep epoch could be been preceded by wake
    sleep_indices = np.where(sleep_wake[:, 1] == sleep_idx)
    sleep_wake = sleep_wake[np.arange(sleep_indices[0][0], sleep_indices[0][-1] + 1), :]

    missing_epochs = np.isnan(sleep_wake[:, 1])
    sleep_wake[missing_epochs, 1] = 0.

    wake_or_missing = (sleep_wake[:, 1] == 0) | missing_epochs
    contiguous_sequences = locate_contiguous(wake_or_missing)
    sequence_length = np.squeeze(np.diff(contiguous_sequences, 1), 1)
    sequences_are_valid_length = sequence_length <= max_sequence_length
    valid_length_sequences = contiguous_sequences[sequences_are_valid_length]
    if np.any(valid_length_sequences):
        # expand, combine, remove sequence positions that were originally missing
        sequences_expanded = [np.arange(x, y) for x, y in valid_length_sequences]
        sequences_combined = np.hstack(sequences_expanded)
        sequence_indices = sequences_combined[np.isin(sequences_combined, np.where(~missing_epochs))]
    else:
        sequence_indices = np.array([])

    # no valid indices if either there were no valid length sequences, or all valid length were composed of missing
    if len(sequence_indices) == 0:
        warnings.warn('There are no Wake sequences which match the criteria, returning NaN')
        return np.nan

    sequence_timestamps = sleep_wake[sequence_indices, 0]

    potential_lag_values = np.arange(-1 * max_lag, max_lag + 1) * epoch_length_seconds
    counts = np.zeros((2, len(potential_lag_values)))

    activity_timestamps = actigraphy[:, 0]
    activity_counts = actigraphy[:, 1]

    # shift the activity counts +- relative to staging references
    for idx, shift in enumerate(potential_lag_values):
        shifted_actigraphy_times = activity_timestamps - shift
        shifted_indices = associate_timestamps(sequence_timestamps, shifted_actigraphy_times, 30)
        shifted_indices = shifted_indices.compressed()
        counts_at_shift = activity_counts[shifted_indices]
        num_above_threshold = np.sum(counts_at_shift >= activity_threshold)
        num_below_threshold = np.sum(counts_at_shift < activity_threshold)
        counts[:, idx] = (num_above_threshold, num_below_threshold)

    # in the event of no actigraphy epochs above threshold (for example due to missing data),
    # the denominator here may be zero.
    with np.errstate(invalid='ignore'):
        proportions = counts[0, :] / np.sum(counts, 0)

    # compare each shifted value to 0
    zero_shift_idx = np.where(potential_lag_values == 0)[0][0]
    zero_shift_proportion = proportions[zero_shift_idx]
    shift_greater_match = proportions > zero_shift_proportion

    p_values = np.ones(len(potential_lag_values))
    for s in range(len(potential_lag_values)):
        if shift_greater_match[s]:
            p_values[s] = stats.boschloo_exact(counts[:, [s, zero_shift_idx]]).pvalue

    # identify if any of the time shifts generated a significant increase in the
    significant_shift = p_values < significance_threshold
    potential_shift_idx = significant_shift & shift_greater_match
    if np.any(potential_shift_idx):
        subset_values = potential_lag_values[potential_shift_idx]
        subset_coefs = proportions[potential_shift_idx]
        lag_value = subset_values[np.argmax(subset_coefs)]
    else:
        lag_value = 0

    return lag_value


def identify_sleep_interval(sleep_wake, wake_epoch_threshold=120):
    sleep_epoch_indices = np.squeeze(np.where(sleep_wake == 1))
    sleep_epoch_differences = np.diff(sleep_epoch_indices, prepend=0)
    sleep_onsets = sleep_epoch_indices[sleep_epoch_differences >= wake_epoch_threshold]
    # identify the length of each sleep period
    num_sleep_intervals = len(sleep_onsets)
    sleep_intervals = np.zeros([num_sleep_intervals, 2], dtype='int')
    for idx, onset in enumerate(sleep_onsets):
        sleep_intervals[idx, 0] = onset
        if idx < (num_sleep_intervals - 1):
            offset = np.max(sleep_epoch_indices[sleep_epoch_indices < sleep_onsets[idx + 1]])
        else:
            offset = np.max(sleep_epoch_indices)
        sleep_intervals[idx, 1] = offset
    interval_length = (sleep_intervals[:, 1] - sleep_intervals[:, 0])
    primary_interval = sleep_intervals[np.argmax(interval_length), :]
    return primary_interval
