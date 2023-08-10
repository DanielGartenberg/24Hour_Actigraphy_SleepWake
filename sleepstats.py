import numpy as np


def total_sleep_time(sleep_wake_classifications, epoch_length_seconds):
    epochs_classed_sleep = sum([x is True for x in sleep_wake_classifications])
    return (epochs_classed_sleep * epoch_length_seconds) / 60


def sleep_efficiency(sleep_wake_classifications):
    epochs_classed_sleep = sum([x is True for x in sleep_wake_classifications])
    epochs_classed_wake = sum([x is False for x in sleep_wake_classifications])
    return (epochs_classed_sleep / (epochs_classed_sleep + epochs_classed_wake)) * 100


def sleep_onset_latency(sleep_wake_classifications, epoch_length_seconds):
    # sleep onset latency in minutes, according to AASM criteria
    sleep_epoch_indices = [i for i, v in enumerate(sleep_wake_classifications) if v is True]
    if any(sleep_epoch_indices):
        first_sleep_epoch_index = min(sleep_epoch_indices)
        return (first_sleep_epoch_index * epoch_length_seconds) / 60
    else:
        return None


def wake_after_sleep_onset_including_twak(sleep_wake_classifications, epoch_length_seconds):
    # this definition of WASO includes the wake following the final sleep epoch
    # which is sometimes considered separately as 'TWAK'
    sleep_epoch_indices = [i for i, v in enumerate(sleep_wake_classifications) if v is True]
    wake_epoch_indices = [i for i, v in enumerate(sleep_wake_classifications) if v is False]
    if any(sleep_epoch_indices):
        first_sleep_epoch_index = min(sleep_epoch_indices)
        last_sleep_epoch_index = max(sleep_epoch_indices)
        wake_epochs_after_sleep = [x for x in wake_epoch_indices if
                                   (x > first_sleep_epoch_index) and (x < last_sleep_epoch_index)]
        return (len(wake_epochs_after_sleep) * epoch_length_seconds) / 60
    else:
        return None


def waso(sleep_wake_classifications, epoch_length_seconds):
    # Wake after sleep onset, this defintiion excludes wake after the final sleep epoch
    # Wake after the final sleep epoch is instead considered TWAK
    sleep_wake_classifications = np.array(sleep_wake_classifications)
    sleep_epoch_indices = np.where(sleep_wake_classifications == True)[0]
    if any(sleep_epoch_indices):
        first_sleep_epoch_index = np.min(sleep_epoch_indices)
        last_sleep_epoch_index = np.max(sleep_epoch_indices)
        waso_epochs = sleep_wake_classifications[first_sleep_epoch_index:last_sleep_epoch_index] == False
        return (np.sum(waso_epochs) * epoch_length_seconds) / 60
    else:
        return None


def number_awakenings(sleep_wake_classifications):
    sleep_wake_np = np.array([np.nan if x is None else x for x in sleep_wake_classifications])
    sleep_wake_differences = np.diff(sleep_wake_np.astype(float))
    return np.sum(sleep_wake_differences == -1).tolist()


def number_awakenings_smoothed(sleep_wake_classifications, epoch_reset_point):
    # convert to numpy, trim trailing awakenings, find awakenings, only keep awakenings
    sleep_wake_np = np.array([np.nan if x is None else x for x in sleep_wake_classifications])
    sleep_wake_trimmed = np.trim_zeros(sleep_wake_np, trim='fb')
    wake_indices = np.where(sleep_wake_trimmed == 0)
    wake_index_diff = np.diff(np.insert(wake_indices, 0, (-1 * epoch_reset_point)))
    return np.nansum(wake_index_diff >= epoch_reset_point).tolist()
