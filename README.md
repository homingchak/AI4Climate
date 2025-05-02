This project uses the Hurricanes and Typhoons dataset from NOAA, which can be accessed at https://www.kaggle.com/datasets/noaa/hurricane-database/data

The contents inside the .zip file should be extracted to the same directory as the programs. 

A copy of both Atlantic and Pacific data (atlantic.csv and pacific.csv) is already put in the main branch.


This project aims to explore the use of random forest classifiers and XGBoost algorithm to predict rapid intensification of tropical cyclones.


utils.py contains functions used in the other programs. 

It does not need to be run.

hurricane_analysis.py goes through both Atlanta and Pacific datasets and generates brief summaries, as well as producing respective plots of hurricanes on a map, histograms of wind speed distribution, and graphs of hurricane frequency over time.



*** Before running RI_prediction_random_forest_v1.py, RI_prediction_random_forest_v2.py, RI_prediction_random_forest_v3.py, or RI_prediction_xgboost.py, select which dataset you want the model to run on (atlantic.csv / pacific.csv) by commenting out the appropriate line at the start of main(), and commenting out the approrpiate group of parameters in param_grid. The 2 groups of parameters are optimized based on the labelled dataset. ***



RI_prediction_random_forest_v1.py trains the RF V1 model with only 4 features and evaluates it on test set, then outputs a brief summary and a classification report including F1 score and other evaluation criteria, with a plot of feature importance.

RI_prediction_random_forest_v2.py trains the RF V2 model with 4 extra features and evaluates it on test set, then outputs a brief summary and a classification report including F1 score and other evaluation criteria, with a plot of feature importance.

RI_prediction_random_forest_v3.py trains the RF V3 model with all features and evaluates it on test set, then outputs a brief summary and a classification report including F1 score and other evaluation criteria, with a plot of feature importance.

RI_prediction_xgboost.py trains the XGBoost model with all features and evaluates it on test set, then outputs a brief summary and a classification report including F1 score and other evaluation criteria, with a plot of feature importance.
