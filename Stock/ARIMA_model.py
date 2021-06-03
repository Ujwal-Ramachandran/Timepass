import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Location of CSV that contains stats")
data.head()

# Check for null if yes drop that
if data.isnull().values.any() :
  data.dropna(subset = ['Close'], inplace=True)

plt.figure()
lag_plot(data["Open"], lag=3)
plt.title("Your Company's(Comp) Stock - Autocorrelation plot with lag = 3")
plt.show()

plt.plot(data["Date"], data["Close"])
plt.xticks(np.arange(0,1259, 200), data["Date"][0:1259:200], rotation = 45)
plt.title("Comp stock price over time")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

data = data.dropna(axis=0,how='any')
data_test = pd.read_csv("Test data Location")
data_test = data_test.dropna(axis=0,how='any')
train_data, test_data = data,data_test
training_data = train_data['Close'].values
test_data = test_data['Close'].values
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(4,1,0)) #ARIMA Parameters p,d and q
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)
MSE_error = mean_squared_error(test_data, model_predictions)
print("Testing Mean Squared Error is {}".format(MSE_error))

test_set_range = data_test.index
plt.plot(test_set_range, model_predictions, color="Orange", marker="*", linestyle="dashed",label="Predicted Price")
plt.plot(test_set_range, test_data, color="Blue", label="Actual Price")
# Predicted graph usong training set superimposed with graph from test set 
plt.title("Comp Prices Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
d = len(test_set_range)
plt.xticks(np.arange(0,d,3),data_test.Date[0:d:3], rotation = 45)
plt.legend()
plt.show()

