# San Francisco Crime Prediction with XGBoost

### Overview
This project aims to predict crime categories in San Francisco using machine learning techniques. The dataset contains information about various incidents reported in San Francisco, including details such as time, location, and type of crime. Our goal is to build a predictive model using XGBoost, a popular gradient boosting algorithm, to classify crimes into multiple categories based on historical data. By leveraging machine learning, we aim to contribute to crime prevention efforts and enhance public safety measures in urban areas.

### Dependencies
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
xgboost
joblib
pickle

### Dataset
The dataset used for this analysis contains information about various incidents reported in San Francisco. It includes details such as time, location, and type of crime.

### Files
SanFranciscoCrimePrediction.ipynb: Jupyter notebook containing the code for data preprocessing, exploratory data analysis, and model building.
Training_Data.csv: CSV file containing the training data.
Test_Data.csv: CSV file containing the test data.
xgb_model.pkl: Pickle file containing the trained XGBoost model.

### Usage
Install the required dependencies.
Run the Jupyter notebook SanFranciscoCrimePrediction.ipynb to train the model and make predictions.
The trained model will be saved as xgb_model.pkl.
The predictions will be saved in predictions.csv.

### Model
The predictive model is built using XGBoost, a popular gradient boosting algorithm. The model is trained to classify crimes into multiple categories based on historical data. The performance of the model is evaluated using accuracy metrics such as precision, recall, and F1-score.

### Results
The model's performance is evaluated using a test dataset. The results are reported in terms of accuracy, precision, recall, and F1-score.

### Conclusion
This project demonstrates the application of machine learning techniques in predicting crime categories in San Francisco. The XGBoost model shows promising results in classifying crimes based on historical data, which can be useful for crime prevention efforts and enhancing public safety measures in urban areas.
