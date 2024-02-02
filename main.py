import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

#data upload
st.title('Sales prediction')
st.subheader('Upload CSV Sales data')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    sales_data = pd.read_csv(uploaded_file)
    st.write(sales_data)

sales_data = sales_data.loc[sales_data['Sales'] >= 0]
sales_data['date'] = pd.to_datetime(sales_data["date"])
sales_data['year'] = sales_data['date'].dt.year
sales_data['month'] = sales_data['date'].dt.month
sales_data['day'] = sales_data['date'].dt.day
sales_data['day_of_week'] = sales_data['date'].dt.day_of_week
sales_data.drop(['date'], axis=1, inplace=True)
sales_data.sort_values(by=['year','month','day'],ignore_index=True)
sales_data.drop(['Province_Route_Customer'], axis=1, inplace=True)
sales_data.drop(['PPG_flavor'], axis=1, inplace=True)


#Discribing Data
st.subheader('Sales Data')
st.write(sales_data)
st.write(sales_data.describe())

#visualizations
sales_df = sales_data.groupby(["year", "month", "day","day_of_week"],as_index=False)["Sales"].sum()
ma14 = sales_df.Sales.rolling(14).mean()
ma7 = sales_df.Sales.rolling(7).mean()
st.subheader('Original Sales chart')
#st.line_chart(sales_df,x=[index],y=['Sales'])
fig = plt.figure(figsize=(20,8))
#plt.style.use('ggplot')
plt.plot(sales_df.Sales,'b',label='Sales')
plt.plot(ma14,'g',label="ma14 days")
plt.plot(ma7,'r',label='ma7 days')
plt.legend(fontsize=15,loc='upper left')
plt.title('Plot Actual vs Moving average 14 days vs Moving average 7 days',fontsize=15,loc='center')
st.pyplot(fig)

#splitting Data into training and testing
data_training = pd.DataFrame(sales_df['Sales'][0:int(len(sales_df)*0.7)])
data_testing = pd.DataFrame(sales_df['Sales'][int(len(sales_df)*0.7):int(len(sales_df))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
#splitting data into x_trand and y_train
#x_train = []
#y_train = []
long = 14
#for i in range (long,data_training_array.shape[0]):
#    x_train.append(data_training_array[i-long:i])
#    y_train.append(data_training_array[i,0])

#x_train, y_train = np.array(x_train), np.array(y_train)

#Load model
model=load_model('keras_model2.h5')

#Tasting part
past_100_days = data_training.tail(len(sales_df)-1)
final_df = past_100_days._append(data_testing,ignore_index=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(long, input_data.shape[0]):
    x_test.append(input_data[i-long:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Making Prediction
y_prediction = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_prediction = y_prediction*scale_factor
y_test = y_test*scale_factor

#final chart
st.subheader('Original Sales vs Prediction')
#plt.style.use('ggplot')
fig2 = plt.figure(figsize=(20,8))
#plt.style.use('ggplot')
#plt.plot(sales_df.Sales,'gray',label='Sales')
plt.plot(y_test,'b',label = 'Original Sales')
plt.plot(y_prediction,'r',label = 'Predicted Sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend(fontsize=15,loc='upper left')
plt.title('Plot Original Sales vs Predicted Sales',fontsize=15,loc='center')
st.pyplot(fig2)

period_predict = st.slider('How many days ahead do you want to forecast?', 1, 365, 30)
st.write("Predict next ", period_predict, 'days')
start_idx = input_data.shape[0] - period_predict
x_test2 = []
y_test2 = []

for i in range(long, input_data.shape[0]):
    x_test2.append(input_data[i-long:i])
    y_test2.append(input_data[i,0])
x_test2, y_test2 = np.array(x_test2), np.array(y_test2)
# Make predictions with our LSTM model
model_preds = model.predict(x_test2)
future_preds = model_preds[-period_predict:]
y_test2 = y_test2*scale_factor
model_preds = model_preds*scale_factor
future_preds = future_preds*scale_factor
final_forecast = np.append(model_preds,future_preds)

st.subheader('Original Sales vs Future Prediction')
#plt.style.use('ggplot')
fig3 = plt.figure(figsize=(20,8))
#plt.style.use('ggplot')
plt.plot(y_test2,'blue',label = 'Original Sales')
# Combine y_test2 and future_preds for a continuous plot
combined_data = model_preds.tolist() + future_preds.tolist()
# Generate correct x-axis values for continuous plotting
x_values = range(len(combined_data))
# Plot the combined data in blue
plt.plot(x_values[:len(y_test2)],combined_data[:len(y_test2)], color='red', label='Predicted Sales')
# Plot final_forecast in red
plt.plot(x_values[len(y_test2):],combined_data[len(y_test2):], color='black', label='Future Predictions')
# Indicate the start of future_preds with a vertical line
plt.axvline(x=len(y_test2), color='green', linestyle='dotted', label='Future Predictions Start')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend(fontsize=15,loc='upper center')
plt.title('Plot Original Sales vs Predicted Sales vs Future Predictions',fontsize=15,loc='center')
st.pyplot(fig3)

future_preds_data = pd.DataFrame(future_preds)
future_preds_data.index += 1
future_preds_data = future_preds_data.set_axis(['Predicted Sales'], axis=1)
st.subheader('Future Predicted Sales')
st.write(future_preds_data)
