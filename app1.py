import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
import yfinance as yf


def _connect(ticker, start, end):
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start=start, end=end)
    return df

@st.cache_data(show_spinner=False)
def get_data(ticker, start, end):
    cache_key = f"{ticker}_{start}_{end}"
    if cache_key not in YahooFinanceConnection.data_cache:
        df = _connect(ticker, start, end)
        if df is not None:
            YahooFinanceConnection.data_cache[cache_key] = df
        else:
            st.warning(f"Invalid stock ticker: {ticker}. Please enter a valid ticker.")
            return None
        
    return YahooFinanceConnection.data_cache[cache_key]


class YahooFinanceConnection:
    data_cache = {}

    def cursor(self, ticker, start, end):
        return get_data(ticker, start, end)

# Streamlit app
def main():
    st.title('Stock Minder: Insightful Stock Price Forecasting System')

    st.markdown('Enter Stock Ticker (Please enter correct stock ticker otherwise you will get error, you can find a list of stock tickers/symbols [here](https://stockanalysis.com/stocks/))')

    # Get user input for stock ticker
    user_input = st.text_input('', 'TSLA')

    # Create the custom Connection instance
    connection = YahooFinanceConnection()

    # Retrieve data using the custom Connection
    start = '2010-01-01'
    end = '2023-04-30'
    df = connection.cursor(user_input, start, end)

    if df is not None :

        st.subheader('Data from 2010-2022')
        st.write(df.describe())

        # Plot Closing Price vs Time chart with 100MA & 200MA
        st.subheader('Closing Price vs Time chart with 100MA & 200MA')
        ma100 = df['Close'].rolling(100).mean()
        ma200 = df['Close'].rolling(200).mean()
        fig = plt.figure(figsize=(20, 10))
        plt.plot(ma100, 'r', label='100MA')
        plt.plot(ma200, 'g', label='200MA')
        plt.plot(df['Close'], 'b', label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # The app will now use your custom Connection to retrieve stock data from Yahoo Finance API
        # and display the results based on user input.

        #training process starts
        df1 = df.reset_index()['Close']
        data = df.filter(['Close'])

        df1 = df1[0:len(df1) - 30]

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

        ##splitting dataset into train and test split
        training_size = int(len(df1) * 0.70)
        test_size = len(df1) - training_size
        train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

        # convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        # reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)

        # reshape input to be [samples, time steps, features] which is required for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # loading the model
        model = tf.keras.models.load_model(r'my_model.h5')

        # training process ends

        # testing process starts
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        ##Transform back to original form
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
        original_ytest = scaler.inverse_transform(ytest.reshape(-1, 1))

        ### Plotting
        train = data[:training_size]
        valid = data[training_size + 100 + 1:len(data) - 30]

        valid['Prediction'] = test_predict

        # visualize
        st.subheader('LSTM Model Result')
        fig2 = plt.figure(figsize=(16, 6))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price ', fontsize=18)
        plt.plot(valid[['Close', 'Prediction']])
        plt.legend(['Observation', 'Prediction'], loc='lower right')
        st.pyplot(fig2)

        # predicting stocks for the next 30 days
        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        # demonstrate prediction for the next 30 days
        lst_output = []
        n_steps = 100
        i = 0
        while i < 30:
            if len(temp_input) > 100:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        st.subheader('Future 30 days Stock Price Prediction')

        df2 = scaler.fit_transform(np.array(data).reshape(-1, 1))
        observed = scaler.inverse_transform(df2[len(df2) - 30:])
        observed.tolist()

        # plot
        fig3 = plt.figure(figsize=(16, 6))
        plt.xlabel('no. of days', fontsize=18)
        plt.ylabel('Stock Price ', fontsize=18)
        plt.plot(observed)
        plt.plot(scaler.inverse_transform(lst_output))
        plt.legend(['Observed', 'Predicted'], loc='lower right')
        st.pyplot(fig3)
    
    else:
        st.warning("Please enter a valid stock ticker.")

# Run the Streamlit app
if __name__ == '__main__':
    main()

