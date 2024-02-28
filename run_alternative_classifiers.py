# Use the PyActigraphy project to run alternative sleep/wake classifiers (Sadeh and Scripps)
# this script  uses an alternative Conda environment (described in PyActigraphy.yaml) to not interfere with the
# Conda environment created for the primary analyses

import pandas as pd
import numpy as np
import pyActigraphy
import pathlib

# create new output folders to supplement the original output with the additional columns
original_output_path = 'output'
additional_output_path = 'output_with_additional_classifiers'

# although the additional classifier output won't differ by TCN model type, append them to each output separately
# to conform to how the rest of the processing pipeline is performed
tcn_model_prefix = ['24_Hour_09_10_2023T18_38_25', '24_Hour_Bidirectional_09_09_2023T23_37_28']

for m in tcn_model_prefix:

    input_prediction_directory = pathlib.Path(original_output_path) / m / 'predictions'
    output_prediction_directory = pathlib.Path(additional_output_path) / m / 'predictions'

    if not output_prediction_directory.exists():
        output_prediction_directory.mkdir(parents=True, exist_ok=True)

    prediction_files = input_prediction_directory.glob('*.csv')

    for f in prediction_files:

        data = pd.read_csv(f)

        # assemble a PyActigraphy data object to run alternative classifiiers
        # the actual dates aren't relevant for the classifiers as long as the epoch interval (30-seconds) is correct
        d = {'Activity': data['actigraphy_activity'].to_numpy(copy=True),
             'Offwrist': data['actigraphy_offwrist'].to_numpy(copy=True),
             'Light': np.nan}

        N = len(d['Activity'])
        index = pd.date_range(start='01-01-2020', freq='30s', periods=N)
        pyActData = pd.DataFrame(index=index, data=d)

        pyActRaw = pyActigraphy.io.BaseRaw(
            name="Name",
            uuid='DeviceId',
            format='Pandas',
            axial_mode=None,
            start_time=pyActData.index[0],
            period=(pyActData.index[-1]-pyActData.index[0]),
            frequency=pyActData.index.freq,
            data=pyActData['Activity'],
            light=pyActData['Light']
        )

        # use the off-wrist data to add a data mask
        # in PyActigarphy, off-wrist is implemented as a binary mask, with 1 = the data *passes* through the mask
        # Although, I believe that the sleep scoring functions in PyActigraphy don't incorporate the off-wrist / mask
        pyActRaw.mask = np.abs(pyActData['Offwrist']-1)

        # make and store the new classifications, using the Scripps Clinic and Sadeh algorithms with default parameters
        scripps_predictions = pyActRaw.Scripps()
        sadeh_predictions = pyActRaw.Sadeh()

        data['scripps_classification'] = scripps_predictions.to_numpy()
        data['sadeh_classification'] = sadeh_predictions.to_numpy()

        # generate new name to avoid confusion with previosu version of files
        f_filepath = pathlib.Path(f)
        output_filename = f_filepath.stem + '_additional_classifiers.csv'
        output_filepath = output_prediction_directory / output_filename
        data.to_csv(output_filepath, index=False)
