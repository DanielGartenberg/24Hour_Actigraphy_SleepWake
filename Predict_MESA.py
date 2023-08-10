# Use pre-trained classifiers to generate predictions on MESA dataset
import tensorflow as tf
from tensorflow import keras
from tcn import TCN
import numpy as np
import pandas as pd
import pathlib


# custom layer to attach a mask prior to evaluation
class AttachMask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttachMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, mask):
        outputs = inputs
        outputs._keras_mask = mask
        return outputs

    def get_config(self):
        config = {}
        base_config = super(AttachMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


paradigms = ['24_Hour_08_28_2022T05_10_00', '24_Hour_Bidirectional_09_08_2022T04_59_52']
dataset_path = pathlib.Path('aligned_data/')

for paradigm_name in paradigms:

    output_path = pathlib.Path('output/') / paradigm_name
    prediction_path = output_path / 'predictions'

    data_folds = pd.read_csv('data_folds.csv')
    data_folds = data_folds.loc[data_folds['data_set'] == 'MESA-COMMERCIAL-USE', :]
    data_folds = data_folds.loc[~data_folds['reject_recording']]
    data_folds = data_folds.reset_index(drop=True)

    x_data = []
    y_data = []

    for d in data_folds['csv_filename']:
        data = pd.read_csv(dataset_path / d)
        off_wrist = data['actigraphy_offwrist'].values
        activity = data['actigraphy_activity'].values
        activity[off_wrist] = np.NaN
        x_data.append(activity)
        y_data.append(data['sleep_wake'].values)

    # store data lengths to later remove padded values from output
    data_lengths = [len(d) for d in x_data]

    # pad sequences up to maximum length with NaN
    max_len = np.max(data_lengths)
    x_data = keras.preprocessing.sequence.pad_sequences(x_data, maxlen=max_len, padding='post', dtype='float16', value=np.NaN)
    y_data = keras.preprocessing.sequence.pad_sequences(y_data, maxlen=max_len, padding='post', dtype='float16', value=np.NaN)

    x_stacked = np.expand_dims(np.vstack(x_data), 2)
    y_stacked = np.expand_dims(np.vstack(y_data), 2)

    # mask values that were missing either x or y
    value_mask = ~np.isnan(x_stacked) & ~np.isnan(y_stacked)

    # replace NaN values with valid values
    y_stacked[np.isnan(y_stacked)] = 0
    x_stacked[np.isnan(x_stacked)] = 0

    for f in np.arange(5):

        model_path = f'output/{paradigm_name}/models/{paradigm_name}_Fold_{f}'
        model = tf.keras.models.load_model(model_path)

        test_rows = np.isin(data_folds['fold'], f)

        x_test = x_stacked[test_rows, :, :]
        y_test = y_stacked[test_rows, :, :]
        mask_test = value_mask[test_rows, :, :]

        # apply the model with best parameters to the test fold
        predictions = model.predict((x_test, mask_test))
        classifications = predictions[:, :, 0] > 0.5

        # reduce data to non-padded values, save predictions per recording
        for idx, d in enumerate(data_folds['csv_filename'][test_rows].values):
            data = pd.read_csv(dataset_path / d)
            num_rows = data.shape[0]
            data['probability_sleep'] = predictions[idx, 0:num_rows, :]
            data['classification_sleep'] = classifications[idx, 0:num_rows]
            data.to_csv(prediction_path / (paradigm_name + '_Predictions_' + d))
