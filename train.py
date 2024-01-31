import os
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.keras import log_model
from mlflow.models import infer_signature
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_log_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import main

logging.basicConfig(level=logging.warn)
logger = logging.getLogger(__name__)
# Specify tracking server
mlflow.set_tracking_uri(uri="http://localhost:5000")
database = '/Users/mamon/Documents/GitHub/65130468_BD528-Y2023_Final_Project-/data/Candy_data_22_23.csv'
sales_data = pd.read_csv(database)
sales_data = sales_data.loc[sales_data[Sales] > 0]
sales_data[date] = pd.to_datetime(sales_data["date"])
sales_data[year] = sales_data[date].dt.year
sales_data[month] = sales_data[date].dt.month
sales_data[day] = sales_data[date].dt.day
sales_data[day_of_week] = sales_data[date].dt.day_of_week
sales_data.drop([date], axis=1, inplace=True)
sales_data.sort_values(by=[year,month,day],ignore_index=True)
sales_data.drop([Province_Route_Customer], axis=1, inplace=True)
sales_data.drop([PPG_flavor], axis=1, inplace=True)
sales_df = sales_data.groupby(["year", "month", "day","day_of_week"],as_index=False)["Sales"].sum()


#splitting Data into training and testing
data_training = pd.DataFrame(sales_df[Sales][0:int(len(sales_df)*0.7)])
data_testing = pd.DataFrame(sales_df[Sales][int(len(sales_df)*0.7):int(len(sales_df))])
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
#splitting data into x_trand and y_train
x_train = []
y_train = []
long = 14
for i in range (long,data_training_array.shape[0]):
    x_train.append(data_training_array[i-long:i])
    y_train.append(data_training_array[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
# Define the model hyperparameters

num_units_lstm1 = 500
num_units_lstm2 = 700
num_units_lstm3 = 900
num_units_lstm4 = 1100
activation = 'relu'
return_sequences = True
dropout_rate1 = 0.2
dropout_rate2 = 0.3
dropout_rate3 = 0.4
dropout_rate4 = 0.5
learning_rate = 0.001  # ตัวอย่างของการกำหนดค่า learning rate
epochs = 200
loss_function = 'mean_squared_error'
# Splitting data into training and testing
data_training = pd.DataFrame(sales_df['Sales'][0:int(len(sales_df)*0.7)])
data_testing = pd.DataFrame(sales_df['Sales'][int(len(sales_df)*0.7):int(len(sales_df))])
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Splitting data into x_train and y_train
x_train = []
y_train = []
long = 14
for i in range (long,data_training_array.shape[0]):
    x_train.append(data_training_array[i-long:i])
    y_train.append(data_training_array[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)

# Define the model
model = Sequential()
model.add(LSTM(units=num_units_lstm1, activation=activation, return_sequences=return_sequences,input_shape = (x_train.shape[1],1)))
model.add(Dropout(dropout_rate1))
model.add(LSTM(units=num_units_lstm2, activation=activation, return_sequences=return_sequences))
model.add(Dropout(dropout_rate2))
model.add(LSTM(units=num_units_lstm3, activation=activation, return_sequences=return_sequences))
model.add(Dropout(dropout_rate3))
model.add(LSTM(units=num_units_lstm4, activation=activation))
model.add(Dropout(dropout_rate4))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam',loss=loss_function)

# Fit the model
model.fit(x_train,y_train,epochs=epochs)

# Evaluate the model
MSE = model.evaluate(x_train, y_train)

# Predict the test set
past_100_days = data_training.tail(len(sales_df)-1)
final_df = past_100_days._append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(long, input_data.shape[0]):
    x_test.append(input_data[i-long:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_prediction = model.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_prediction)


# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "LSTM model for Sales data")

    # Infer the model signature
    signature = infer_signature(x_train, model.predict(x_test))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=LSTM,
        artifact_path="lstm_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="LSTM_Sales_Prediction",
    )

