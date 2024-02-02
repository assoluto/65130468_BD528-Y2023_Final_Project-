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
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
#import MLProject
#from MLProject import parameters
parameters = {
    "n_trials": int(3),
    "data_dir": "./Users/mamon/Documents/GitHub/65130468_BD528-Y2023_Final_Project-/data/",
    "number_of_layers": int(4),
    "num_units_lstm1": int(500),
    "num_units_lstm2": int(700),
    "num_units_lstm3": int(900),
    "num_units_lstm4": int(1100),
    "return_sequences": bool(True),
    "dropout_rate1": float(0.2),
    "dropout_rate2": float(0.3),
    "dropout_rate3": float(0.4),
    "dropout_rate4": float(0.5),
    "learning_rate": float(0.001),
    "batch_size": int(16),
    "epochs": int(200),
    "optimizer": "adam",
    "activation": "relu",
    "loss_function": "mse",
    "embedding_size": int(32)
}
# Configure logging with the correct level
logging.basicConfig(level=logging.INFO)  # Set log level to INFO
logger = logging.getLogger(__name__)

# Specify tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Set MLflow log level (optional)
#mlflow.set_tracking_uri(uri="http://localhost:5000")
""" database = '/Users/mamon/Documents/GitHub/65130468_BD528-Y2023_Final_Project-/data/Candy_data_22_23.csv'
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
sales_df = sales_data.groupby(["year", "month", "day","day_of_week"],as_index=False)["Sales"].sum() """


def load_data(data_dir, filename="Candy_data_22_23.csv"):
    """
    Loads sales data from a CSV file and aggregates it by year, month, day, and day of week.
    Args:
        data_dir (str): The path to the directory containing the CSV file.
        filename (str, optional): The name of the CSV file. Defaults to "Candy_data_22_23.csv".
    Returns:
        pd.DataFrame: A DataFrame containing the aggregated sales data.
    """
    # Load the CSV file
    try:
        sales_data = pd.read_csv(os.path.join(data_dir, filename))
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {os.path.join(data_dir, filename)}")
    # Check for required columns
    required_columns = ["date", "Sales", "Province_Route_Customer", "PPG_flavor"]
    if not all(col in sales_data.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {', '.join(set(required_columns) - set(sales_data.columns))}")
    # Filter data with positive sales (optional)
    sales_data = sales_data.loc[sales_data["Sales"] > 0]
    # Extract date features
    sales_data["date"] = pd.to_datetime(sales_data["date"])
    sales_data["year"] = sales_data["date"].dt.year
    sales_data["month"] = sales_data["date"].dt.month
    sales_data["day"] = sales_data["date"].dt.day
    sales_data["day_of_week"] = sales_data["date"].dt.day_of_week
    # Drop unnecessary columns
    sales_data.drop(["Province_Route_Customer", "PPG_flavor"], axis=1, inplace=True)
    # Aggregate sales by year, month, day, and day of week
    sales_df = sales_data.groupby(["year", "month", "day", "day_of_week"], as_index=False)["Sales"].sum()
    return sales_df

def preprocess_data(sales_df, long=14):
    """Preprocesses the sales data, splits it into training and testing sets,
    and prepares data for prediction.
    Args:
        sales_df (pd.DataFrame): The sales DataFrame.
        long (int): The lookback window for sequence creation.
    Returns:
        tuple: (x_train, y_train, x_test, y_test, scaler)
    """
    # Split data into training and testing
    data_training = sales_df['Sales'][0:int(len(sales_df) * 0.7)]
    data_testing = sales_df['Sales'][int(len(sales_df) * 0.7):int(len(sales_df))]
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))  # Reshape for scaling

    # Create sequences for training
    x_train = []
    y_train = []
    for i in range(long, data_training_array.shape[0]):
        x_train.append(data_training_array[i - long:i])
        y_train.append(data_training_array[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Prepare data for prediction
    past_100_days = data_training.tail(len(sales_df) - 1)
    final_df = past_100_days._append(data_testing, ignore_index=True)
    input_data = scaler.transform(final_df.values.reshape(-1, 1))  # Reshape for scaling
    x_test = []
    y_test = []
    for i in range(long, input_data.shape[0]):
        x_test.append(input_data[i - long:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, y_train, x_test, y_test, scaler

""" #splitting Data into training and testing
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

    x_train, y_train = np.array(x_train), np.array(y_train) """
def build_model(parameters):
    """Builds the LSTM model with the specified hyperparameters.
    Args:
        num_units_lstm1 (int): Number of units in the first LSTM layer.
        num_units_lstm2 (int): Number of units in the second LSTM layer.
        num_units_lstm3 (int): Number of units in the third LSTM layer.
        num_units_lstm4 (int): Number of units in the fourth LSTM layer.
        activation (str): Activation function to use.
        return_sequences (bool): Whether to return sequences from LSTM layers.
        dropout_rate1 (float): Dropout rate for the first LSTM layer.
        dropout_rate2 (float): Dropout rate for the second LSTM layer.
        dropout_rate3 (float): Dropout rate for the third LSTM layer.
        dropout_rate4 (float): Dropout rate for the fourth LSTM layer.
        learning_rate (float): Learning rate for the optimizer.
    Returns:
        tensorflow.keras.models.Sequential: The compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=500, activation='relu', return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))  # Input shape based on training data
    model.add(Dropout(0.2))
    model.add(LSTM(units=700, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=900, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=1100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_log_model(parameters,model,scaler,x_train,y_train,x_test,y_test):  # Add epochs as a parameter
    """Trains the model, evaluates it, and logs results and the model to MLflow.
    Args:
        parameters (dict): Model hyperparameters.
        model (tensorflow.keras.models.Sequential): The compiled LSTM model.
        scaler (sklearn.preprocessing.MinMaxScaler): The fitted scaler.
        epochs (int): Number of epochs to train for.
    """
    # Train the model
    history = model.fit(x_train, y_train, epochs=20)
    # Inverse transform predictions and targets
    y_prediction = scaler.inverse_transform(model.predict(x_test))  # Use inverse_transform
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))  # Reshape for inverse_transform
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate the model
    MSE = model.evaluate(x_train, y_train)
    mse = mean_squared_error(y_test, y_prediction)

    # Log results to MLflow
    with mlflow.start_run():
        mlflow.log_params(parameters)
        mlflow.log_metric("mse", mse)
        mlflow.set_tag("Training Info", "LSTM model for Sales data")
        mlflow.keras.log_model(model, "model")
        mlflow.log_metrics({"mse": history.history["mse"][-1], "mae": history.history["mae"][-1]})

        # Log the model (correcting the model type and signature inference)
        signature = infer_signature(x_train, model.predict(x_test))
        mlflow.keras.log_model(
            model, "lstm_model", signature=signature, input_example=x_test
        )  # Use mlflow.keras.log_model for Keras models

if __name__ == '__main__':

    # Define the parameters for the model
    parameters = {
    "number_of_layers": int(4),
    "num_units_lstm1": int(550),
    "num_units_lstm2": int(700),
    "num_units_lstm3": int(900),
    "num_units_lstm4": int(1100),
    "return_sequences": bool(True),
    "dropout_rate1": float(0.2),
    "dropout_rate2": float(0.3),
    "dropout_rate3": float(0.4),
    "dropout_rate4": float(0.5),
    "learning_rate": float(0.001),
    "batch_size": int(16),
    "epochs": int(200),
    "optimizer": "adam",
    "activation": "relu",
    "loss_function": "mse",
    "embedding_size": int(32)
}

    # Load and preprocess the data
    sales_df = load_data("data")  # Assuming data is in a folder named "data"
    x_train, y_train, x_test, y_test, scaler = preprocess_data(sales_df)

    # Build the model
    model = build_model(parameters)

    # Set the number of epochs
    epochs = int(220)  # Example number of epochs

    # Train and log the model
    train_and_log_model(parameters, model, scaler, x_train, y_train, x_test, y_test)












""" # Define the model hyperparameters
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
y_prediction = model.predict(x_test)
# Calculate metrics
accuracy = accuracy_score(y_test, y_prediction)

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

 """