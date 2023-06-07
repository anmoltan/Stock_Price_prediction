import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import pandas_datareader.data as pdr

import tensorflow as tf


import streamlit as st
import yfinance as yf

yf.pdr_override()

start='2010-01-01'
end= '2023-04-30'

st.title('Stock Minder: Insightful Stock Price Forecasting System')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = pdr.get_data_yahoo(user_input,start,end)

#describe data
st.subheader('data from 2010-2022')
st.write(df.describe())




st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 =df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (20,10))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#training process starts

df1=df.reset_index()['Close']
data= df.filter(['Close'])

df1= df1[0:len(df1)-30]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

#loading the model
model = tf.keras.models.load_model('C:\Users\anmol\OneDrive\Desktop\New folder\stock prediction\my_model.h5')


#training proces ends

#testing process starts
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(ytest.reshape(-1,1))

### Plotting 
train= data[:training_size]
valid= data[training_size+100+1:len(data)-30]

valid['Prediction']= test_predict

#visualize

fig2= plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ', fontsize=18)
#plt.plot(train['Close'])
plt.plot(valid[['Close', 'Prediction']])
plt.legend([ 'Val', 'Prediction'], loc='lower right')
st.pyplot(fig2)

#predicting stocksfor next 30 days
x_input=test_data[len(test_data)-100:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

#print(lst_output)

#day_new=np.arange(1,30)
#day_pred=np.arange(1,30)


st.subheader('Future 30 days Stock Price Prediction')

df2=scaler.fit_transform(np.array(data).reshape(-1,1))
observed=scaler.inverse_transform(df2[len(df2)-30:])
observed.tolist()

#plot
fig3= plt.figure(figsize=(16,6))
plt.plot(observed)
plt.plot(scaler.inverse_transform(lst_output))
st.pyplot(fig3)
