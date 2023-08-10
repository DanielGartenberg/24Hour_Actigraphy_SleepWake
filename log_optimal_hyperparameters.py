import tensorflow as tf
import numpy as np
from tcn import TCN
import pandas as pd

models = ['24_Hour_08_28_2022T05_10_00', '24_Hour_Bidirectional_09_08_2022T04_59_52']

for model_run_name in models:

    model_folds = np.arange(5)
    hyperparam_df = pd.DataFrame({'fold': model_folds + 1, 'nb_filters': np.nan})

    for idx, fold in enumerate(model_folds):
        print(fold)
        model_path = f'output/{model_run_name}/models/{model_run_name}_Fold_{fold}'
        model = tf.keras.models.load_model(model_path)
        model_config = model.get_config()
        tcn_layer = model_config['layers'][2]
        dropout_layer = model_config['layers'][3]

        hyperparam_df.loc[idx, 'nb_filters'] = int(tcn_layer['config']['nb_filters'])
        hyperparam_df.loc[idx, 'kernel_size'] = int(tcn_layer['config']['kernel_size'])
        hyperparam_df.loc[idx, 'dilations'] = np.max(tcn_layer['config']['dilations'])
        hyperparam_df.loc[idx, 'tcn_dropout'] = tcn_layer['config']['dropout_rate']
        hyperparam_df.loc[idx, 'layer_norm'] = tcn_layer['config']['use_layer_norm']
        hyperparam_df.loc[idx, 'use_batch_norm'] = tcn_layer['config']['use_batch_norm']

        hyperparam_df.loc[idx, 'dense_dropout_rate'] = dropout_layer['config']['rate']
        hyperparam_df.loc[idx, 'learning_rate'] = "{:.3e}".format(model.optimizer.lr.numpy())


    hyperparam_df = hyperparam_df.transpose()
    hyperparam_df.to_csv(f'tables/{model_run_name}_Model_Hyperparameters.csv', float_format='%.3f')
