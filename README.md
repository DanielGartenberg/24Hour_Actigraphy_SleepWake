
Code for Roberts DM, Schade MM, Master L, Honavar VG, Nahmod NG, Chang A-M, Garteberg D, & Buxton OM, "Performance of an open machine learning model to classify sleep/wake from actigraphy across ~24-hour intervals without knowledge of rest timing.", currently in-press at *Sleep Health*.


Roberts DM, Schade MM, Master L, et al. Performance of an open machine learning model to classify sleep/wake from actigraphy across ∼24-hour intervals without knowledge of rest timing. Sleep Health. Published online August 10, 2023. doi:10.1016/j.sleh.2023.07.001


The paper is Open Access, available via: [https://www.sciencedirect.com/science/article/pii/S2352721823001341](https://www.sciencedirect.com/science/article/pii/S2352721823001341)


Directory 'environment' contains Conda environment files for the main classifier, as well as for the supplemental classifications (Sadeh and Scripps Clinic) using the PyActigraphy package

Directory 'output' contains trained models for each of the 5 cross-validation folds, for each of the two models. One model operates bidirectionally, and was presented within the main manuscript. The second model operates 'causally', i.e. only on currnet and past epochs, and was presented within the supplemental material.

Directory "statistics\_and_plotting' contains R code for summarizing performance, running inferential statistics, and generating manuscript figues and tables.

The main directory contains Python code for data preprocessing, training the models, and evaluating model performance.

Note that the files containing the actigraphy or summary data are not included.