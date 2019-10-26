[Home](https://mgcodesandstats.github.io/) |
[Portfolio](https://mgcodesandstats.github.io/portfolio/) |
[Terms and Conditions](https://mgcodesandstats.github.io/terms/) |
[E-mail me](mailto:contact@michaeljgrogan.com) |
[LinkedIn](https://www.linkedin.com/in/michaeljgrogan/)

# Part 3: Predicting Weekly Hotel Cancellations with an LSTM Network

*This is Part 3 of a three part project on predicting hotel cancellations with machine learning.*

*[- Part 1: Predicting Hotel Cancellations with Support Vector Machines and ARIMA](https://michael-grogan.com/hotel-cancellations)*

*[- Part 2: Predicting Hotel Cancellations with a Keras Neural Network](https://michael-grogan.com/hotel-cancellations-neuralnetwork)*

As I discussed in a [previous post](https://www.michael-grogan.com/hotel-cancellations/), hotel cancellations can be problematic for businesses in the industry - cancellations lead to lost revenue, and this can also cause difficulty in coordinating bookings and adjusting revenue management practices.

Aside from analyzing which customers are less likely to cancel their bookings and allow hotels to amend their marketing strategy accordingly, it can also be useful to predict fluctuations in cancellations on a week-by-week basis in order for hotel chains to allocate capacity accordingly.

## LSTM (Long-Short Term Memory Network)

LSTMs are sequential neural networks that assume dependence between the observations in a particular series. As such, they have increasingly come to be used for time series forecasting purposes. The Jupyter notebooks with full code, plots, and results can be found [here](https://github.com/MGCodesandStats/hotel-cancellations-lstm).

In the last example, the cancellation data was already sorted into weekly values by pandas. The total weekly cancellations were sorted as follows:

![cancellationweeks.png](cancellationweeks.png)

Now, an LSTM is used to predict cancellations for both the validation and test sets, and ultimately gauge model performance in terms of mean directional accuracy and root mean square error (RMSE).

## Dataset Matrix Formation and Model Configuration

Let’s begin the analysis for the H1 dataset. The first 100 observations from the created time series is called. Then, a dataset matrix is created and the data is scaled.

```
df = df[:100]

# Form dataset matrix
def create_dataset(df, previous=1):
    dataX, dataY = [], []
    for i in range(len(df)-previous-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)
```

The data is then normalized with MinMaxScaler in order to allow the neural network to interpret it properly:

```
# normalize dataset with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
df
```

Here is a sample of the output:

```
array([[0.11782946],
       [0.20465116],
       [0.32093023],
       [0.46511628],
       [0.21395349],
       [0.44496124],
       [0.63100775],
       [0.26356589],
       [0.29612403],
       [0.37984496],
       ...
       [0.51472868],
       [0.40620155],
       [0.31782946],
       [0.6372093 ],
       [0.66046512],
       [0.61395349],
       [0.84806202],
       [0.79224806],
       [0.56434109],
       [1.        ]])
```

The data is partitioned into training and test sets, with the *previous* parameter set to 5:

```
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# Training and Validation data partition
train_size = int(len(df) * 0.8)
val_size = len(df) - train_size
train, val = df[0:train_size,:], df[train_size:len(df),:]

# Number of previous
previous = 5
X_train, Y_train = create_dataset(train, previous)
X_val, Y_val = create_dataset(val, previous)
```

When the *previous* parameter is set to this, this essentially means that the value at time *t* (Y_train for the training data), is being predicted using the values *t-1*, *t-2*, *t-3*, *t-4*, and *t-5* (all under X_train).

Here is a sample of the *Y_train* array:

```
array([0.44496124, 0.63100775, 0.26356589, 0.29612403, 0.37984496,
       0.48062016, 0.63255814, 0.60930233, 0.46976744, 0.57364341,
       0.64031008, 0.2       , 0.27596899, 0.07131783, 0.09302326,
       ...
       0.4248062 , 0.35968992, 0.20310078, 0.19689922])
```

Here is a sample of the *X_train* array:

```
array([[0.11782946, 0.20465116, 0.32093023, 0.46511628, 0.21395349],
       [0.20465116, 0.32093023, 0.46511628, 0.21395349, 0.44496124],
       [0.32093023, 0.46511628, 0.21395349, 0.44496124, 0.63100775],
       ...
       [0.32868217, 0.26976744, 0.4248062 , 0.35968992, 0.20310078]])
```       

150 epochs are run:

```
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# Generate LSTM network
model = tf.keras.Sequential()
model.add(LSTM(4, input_shape=(1, previous)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=150, batch_size=1, verbose=2)
```

Here are some sample results:

```
Train on 74 samples
Epoch 1/150
74/74 - 1s - loss: 0.2013
Epoch 2/150
74/74 - 0s - loss: 0.1040
Epoch 3/150
74/74 - 0s - loss: 0.0571
Epoch 4/150
74/74 - 0s - loss: 0.0446
Epoch 5/150
74/74 - 0s - loss: 0.0426
...
Epoch 146/150
74/74 - 0s - loss: 0.0324
Epoch 147/150
74/74 - 0s - loss: 0.0324
Epoch 148/150
74/74 - 0s - loss: 0.0330
Epoch 149/150
74/74 - 0s - loss: 0.0323
Epoch 150/150
74/74 - 0s - loss: 0.0321
```

## Training and Validation Predictions

Now, let’s generate some predictions.

```
# Generate predictions
trainpred = model.predict(X_train)
valpred = model.predict(X_val)
```

Here is a sample of training and test predictions:

**Training Predictions**

```
>>> trainpred

array([[0.38761035],
       [0.37727284],
       [0.38431337],
...
       [0.27506492],
       [0.33542088],
       [0.29014656]], dtype=float32)
```

**Test Predictions**

```
>>> valpred

array([[0.26687723],
       [0.2980349 ],
       [0.36651152],
       [0.4387382 ],
       [0.45658803],
       [0.43241972],
       [0.40169773],
       [0.3515324 ],
       [0.27032438],
       [0.39643878],
       [0.49958646],
       [0.5353131 ],
       [0.532136  ],
       [0.3223253 ]], dtype=float32)
```

The predictions are converted back to normal values using ```scaler.inverse_transform```, and the training and test score is calculated.

```
import math
from sklearn.metrics import mean_squared_error

# calculate RMSE
trainScore = math.sqrt(mean_squared_error(Y_train[0], trainpred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(Y_test[0], testpred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
```

**Training and Validation Scores**

```
Train Score: 37.01 RMSE
Validation Score: 35.65 RMSE
```

Here is a plot of the predictions:

![h1predictiongraph.png](h1predictiongraph.png)

The test and prediction arrays are reshaped accordingly, and the function for *mean directional accuracy* is defined:

```
import numpy as np

def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))
```

### Model Results

The mean directional accuracy is now calculated:

```
>>> mda(Y_val, predictions)
0.8571428571428571
```

An MDA of **86%** is obtained, meaning that the model correctly predicts the direction of the actual weekly cancellation trends 86% of the time.

As seen above, a validation score of **35.65** RMSE was also obtained. RMSE is a measure of the deviation in cancellations from the actual values, and assumes the same numerical format as the same. The mean weekly cancellations across the validation data was **109**.

## Testing on unseen (test) data

Now that the model has been trained, the next step is to test the predictions of the model on unseen (or test data).

As previously explained, the value at time *t* is being predicted by LSTM using the values *t-1*, *t-2*, *t-3*, *t-4*, and *t-5*.

The last 10 weekly cancellation values in the series are predicted in this case.

```
actual = np.array([[161,131,139,150,157,173,140,182,143,100]])
```

The previously built model is now used to predict each value using the previous five values in the time series:

```
# Test (unseen) predictions
# (t) and (t-5)
>>> XNew

array([[130, 202, 117, 152, 131],
       [202, 117, 152, 131, 161],
       [117, 152, 131, 161, 131],
       [152, 131, 161, 131, 139],
       [131, 161, 131, 139, 150],
       [161, 131, 139, 150, 157],
       [131, 139, 150, 157, 173],
       [139, 150, 157, 173, 140],
       [150, 157, 173, 140, 182],
       [157, 173, 140, 182, 143]])
```

The variables are scaled appropriately, and ```model.predict``` is invoked:

```
Xnew = scaler.fit_transform(Xnew)
Xnew
Xnewformat = np.reshape(Xnew, (Xnew.shape[0], 1, Xnew.shape[1]))
ynew=model.predict(Xnewformat)
```

Here is an array of the generated predictions:

```
array([0.11751928, 0.2840012 , 0.38806236, 0.22630812, 0.22927041,
       0.4725005 , 0.49718988, 0.62252706, 0.47404462, 0.5425472 ],
      dtype=float32)
```

The array is converted back to the original value format:

```
>>> ynew = ynew * np.abs(maxcancel-mincancel) + np.min(tseries)
>>> ynewpd=pd.Series(ynew)
>>> ynewpd

0     38.444008
1     73.072250
2     94.716972
3     61.072090
4     61.688248
5    112.280106
6    117.415497
7    143.485626
8    112.601280
9    126.849823
dtype: float32
```

Here is the calculated **MDA**, **RMSE**, and **MFE (mean forecast error)**.

**MDA**

```
>>> mda(actualpd, ynewpd)
0.8
```

**RMSE**

```
>>> mse = mean_squared_error(actualpd, ynewpd)
>>> rmse = sqrt(mse)
>>> print('RMSE: %f' % rmse)

RMSE: 66.823950
```

**MFE**

```
>>> forecast_error = (ynewpd-actualpd)
>>> mean_forecast_error = np.mean(forecast_error)
>>> mean_forecast_error

-53.43740997314453
```

Here is a plot of the predicted vs actual cancellations per week:

![predicted-vs-test.png](predicted-vs-test.png)

As we can see, the MDA has dropped slightly, and the RMSE has increased to 66. Based on the graph and the mean forecast error, the model has a tendency to underestimate the values for the weekly cancellations; i.e. the forecast is negatively biased.

## H2 results

The same procedure was carried out on the H2 dataset (cancellation data for a separate hotel in Portugal). Here are the results when comparing the predictions to the test set:

1

## Comparison with ARIMA
