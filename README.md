# StockMinder - Stock_Price_prediction

This study employs a deep learning approach to analyse and make predictions on stock 
prices. The primary objective of this research is to construct a proficient deep-learning model 
that can forecast stock prices by leveraging historical data. To accomplish this, Long Short-Term Memory (LSTM) neural networks, which are specifically designed for analysing time_series data, are utilized in the model. The project makes use of the Yahoo Finance API to 
retrieve historical stock data, followed by pre-processing and scaling of the data using the Min 
Max Scaler. The dataset is then divided into specific training and testing sets, and a sliding 
window technique is applied to generate input/output sequences for the LSTM model. The 
model is then trained using the training data and employed to predict stock prices for the 
upcoming 30 days.

## How to run the code?

Step 1: Open any IDE with Python setup already, if not then make it done.                    
Step 2: Clone this repo open in IDE.                     
Step 3: Open the terminal in this repo only.                           
Step 4: Run the requirements.txt file to install all the necessary modules.                       
                                                       pip install -r requirements.txt                            
Step 5: At last, run the app2.py file.                                           
         streamlit run app2.py

  now you will see an interactive interface where you can predict the stock price of any company just by entering the stock ticker.
