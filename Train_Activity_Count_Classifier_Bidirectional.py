# Activity Count Sleep/Wake Classifier
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from tcn import TCN
import numpy as np
import os
import pandas as pd
import datetime
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


# subclass the Keras-Tuner HyperModel to be able to specify a scaling value (scale 0-1) specific to each training fold
# see: https://stackoverflow.com/questions/63767275/how-do-i-pass-multiple-arguments-through-keras-tuner-function
# Ideally the maximum would be automatically calculated within the model from training data
class ScaledHyperModel(kt.HyperModel):

    def __init__(self, scale_val):
        self.scale_value = scale_val

    def build(self, hp):

        # which if any type of normalization should be used
        normalization_type = hp.Choice('norm_type', ['no_norm', 'batch_norm', 'layer_norm'], default='no_norm')
        use_batch_norm = True if normalization_type == 'batch_norm' else False
        use_layer_norm = True if normalization_type == 'layer_norm' else False

        # expand to a sequence, up to the 2^ max value
        max_dilation_base = hp.Choice('dilations', [1, 2, 3, 4, 5], default=3)
        dilation_val = np.power(2, np.arange(0, max_dilation_base+1)).tolist()

        activity_input = keras.Input(shape=(None, 1), name='activity_input')
        mask_input = keras.Input(shape=(None, 1), name='mask_input')
        activity_scaling = keras.layers.Rescaling(scale=self.scale_value)(activity_input)
        tcn_layer = TCN(nb_filters=hp.Choice('nb_filters', [1, 2, 4, 8, 16, 32, 64], default=8),
                        kernel_size=hp.Choice('kernel_size', [1, 3, 5, 7], default=3),
                        dilations=dilation_val,
                        nb_stacks=1,
                        dropout_rate=hp.Float('tcn_dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=.20),
                        return_sequences=True,
                        use_skip_connections=True,
                        use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm, use_weight_norm=False,
                        padding='same')
        tcn = tcn_layer(activity_scaling)
        dropout_layer = keras.layers.Dropout(hp.Float('dense_dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=.20))(tcn)
        output = keras.layers.Dense(1, activation='sigmoid')(dropout_layer)
        masked_output = AttachMask()(output, mask=mask_input)
        model = keras.Model(inputs=[activity_input, mask_input], outputs=masked_output, name="activity_classifier")

        model.compile(
            loss=[keras.losses.BinaryCrossentropy()],
            optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=.001)),
            metrics=["accuracy"])

        return model



# path to data that has been time aligned between PSG staging and actigraphy in a prior step
paradigm_name = f'24_Hour_Bidirectional_{datetime.datetime.now().strftime("%m_%d_%YT%H_%M_%S")}'
dataset_path = pathlib.Path('aligned_data/')
output_path = pathlib.Path('output/') / paradigm_name
prediction_path = output_path / 'predictions'
model_path = output_path / 'models'
tuner_log_path = output_path / 'keras_tuner_logs'

os.makedirs(prediction_path)
os.makedirs(model_path)
os.makedirs(tuner_log_path)

batch_size = 32
num_epochs = 300

data_folds = pd.read_csv('data_folds.csv')
data_folds = data_folds.loc[data_folds['data_set'] != 'MESA-COMMERCIAL-USE', :] # exclude MESA, not 24 hour
data_folds = data_folds.loc[~data_folds['reject_recording']] # exclude rejected recordings

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

data_folds['fold'] = data_folds['fold'].astype('int')
fold_labels = np.unique(data_folds['fold'])
num_folds = len(fold_labels)

for test_fold in fold_labels:

    # use the next fold as validation, this allows each fold to be validation at one point
    validation_fold = (test_fold + 1) % (num_folds)
    train_folds = np.setdiff1d(fold_labels, [test_fold, validation_fold], assume_unique=True)

    train_rows = np.isin(data_folds['fold'], train_folds)
    validation_rows = np.isin(data_folds['fold'], validation_fold)
    test_rows = np.isin(data_folds['fold'], test_fold)

    x_train = x_stacked[train_rows, :, :]
    y_train = y_stacked[train_rows, :, :]
    mask_train = value_mask[train_rows, :, :]

    x_validation = x_stacked[validation_rows, :, :]
    y_validation = y_stacked[validation_rows, :, :]
    mask_validation = value_mask[validation_rows, :, :]

    x_test = x_stacked[test_rows, :, :]
    y_test = y_stacked[test_rows, :, :]
    mask_test = value_mask[test_rows, :, :]

    # normalize to the maximum non-masked value in the training set for the fold
    max_activity = np.max(x_train[mask_train])
    scale_value = 1./max_activity  # value to scale training data to 0,1

    tf.keras.backend.clear_session()

    hyper_model = ScaledHyperModel(scale_val=scale_value)
    model_tuner = kt.BayesianOptimization(
        hyper_model,
        'val_loss',
        max_trials=100,
        directory=tuner_log_path,
        project_name=(paradigm_name + f'_Fold_{test_fold}'),
        overwrite=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0, restore_best_weights=True)
    model_tuner.search((x_train, mask_train), y_train, batch_size=batch_size, epochs=num_epochs, validation_data=((x_validation, mask_validation), y_validation), callbacks=[early_stopping_callback])
    best_models = model_tuner.get_best_models(1) # get the best model identified
    model = best_models[0]
    model.save(model_path / (paradigm_name + f'_Fold_{test_fold}'))

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
