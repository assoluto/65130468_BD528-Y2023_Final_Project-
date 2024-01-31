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
st.subheader('Sales vs Time chart')
#st.line_chart(sales_df,x=[index],y=['Sales'])
fig = plt.figure(figsize=(20,12))
#plt.style.use('ggplot')
plt.plot(sales_df.Sales,'gray',label='Sales')
plt.plot(ma14,'blue',label='14 days moving avg')
plt.plot(ma7,'green',label='7 days moving avg')
plt.legend()
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
st.subheader('Predicted vs Original Sales')
#plt.style.use('ggplot')
fig2 = plt.figure(figsize=(20,12))
plt.plot(y_test,'gray',label = 'Original Sales')
plt.plot(y_prediction,'red',label = 'Predicted Sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
st.pyplot(fig2)