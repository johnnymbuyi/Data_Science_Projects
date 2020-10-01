import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf




st.write('# Time Series Forecast')
st.write('#### SARIMA vs RNN (LSTM) Model: which one is best suited for this particular sales orders data?')
'''

'''
st.write('SARIMA (Seasonal Autoregressive Integrated Moving-Average) and RNN (Recurrent Neural Network) models are adopted and applied to this time series prediction problem. This process can be applied to business situations, such as: budgets and demand planning.')
'''

'''

# reuse this data across runs
#read_and_cache_csv = st.cache(pd.read_csv)
url = ('/Users/jonathanmbuyi/documents/sales_orders.csv')

@st.cache
def df_load_n_convert():
    raw_data = pd.read_csv(url)
    raw_data.DATE_CREATED = [datetime.strptime(x, '%d/%m/%Y %H:%M:%S') 
                      for x in raw_data.DATE_CREATED
                     ]
    return raw_data


raw_data = df_load_n_convert()
data = raw_data[['DATE_CREATED', 'SORDER_ID']].groupby('DATE_CREATED').count().reset_index()
#st.write(data)

# Date Parameter?
#date_range = st.sidebar.date_input('Select Range', [data.DATE_CREATED.min(), data.DATE_CREATED.max()])
#st.write('Selected Range: ', np.min(date_range).strftime('%d/%m%/%Y'),
#         ' - ', np.max(date_range).strftime('%d/%m%/%Y'))



# Select Date range to use
data = data.loc[(data.DATE_CREATED >= '2008-04-01') & (data.DATE_CREATED < '2019-04-01')]
data.set_index('DATE_CREATED', inplace=True)
data.columns = ['Total_Orders']
df = data.Total_Orders


#Split the dataset into training and validation sets
split_time = int(df.shape[0] * 0.9)
train = df[:split_time]
test  = df[split_time:]
print(train.shape)
print(test.shape)


#------------------------------------------------
# SARIMA MODEL
#------------------------------------------------

# Apply SARIMA Model
@st.cache(allow_output_mutation=True)
def s_model(p, d, q, s):
    sarima_model = SARIMAX(train
                           , order=(p, d, q)
                           , seasonal_order=(p, d, q, s)
                          ).fit(disp=-1)
    return sarima_model


sarima_model = s_model(1, 1, 1, 7)

# Forecast
fc = sarima_model.get_forecast(len(test), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series    = pd.Series(fc.predicted_mean.values,  index=test.index)


#------------------------------------------------
# RNN Model
#------------------------------------------------
split_time = int(data.shape[0] * 0.9)
time_train = data.index[:split_time]
x_train = data.Total_Orders.values[:split_time]
time_valid = data.index[split_time:]
x_valid = data.Total_Orders.values[split_time:]
series = data.Total_Orders.values
series_time = data.index

window_size = 60 


# create time window function
@st.cache
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1) #expanding dimensions of the serie, this is because input shape will be specified on the conv1D
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True) # truncate to remove remainder to have equal size data
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.map(lambda w: (w[:-1], w[1:])) # split into inputs and target
    ds = ds.shuffle(shuffle_buffer) # shuffle ahead of training to prevent sequences bias
    return ds.batch(batch_size).prefetch(1) #batching data into smaller sets/clusters


# create prediction function
#@st.cache(allow_output_mutation=True)
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/Users/jonathanmbuyi/documents/rnn_model.hdf5')
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[:, -1, 0]
    return rnn_forecast


rnn_forecast = load_model()

rnn_fc = pd.Series(rnn_forecast[split_time - window_size:-1], index=time_valid)

rnn_pred = rnn_forecast[:split_time - window_size]
rnn_pred = pd.Series(rnn_pred, index=time_train[window_size:split_time])


#------------------------------------------------
# Plot Forecast against Actuals
#------------------------------------------------
#PLOT PARAMETERS
model_name = st.sidebar.selectbox('Select Forecast Model', ('SARIMA','RNN'))

def get_model(mod_name):
    if mod_name == 'SARIMA':
        train_fc = sarima_model.fittedvalues
        test_fc = fc_series
    else:
        train_fc = rnn_pred
        test_fc = rnn_fc
    return train_fc, test_fc


train_fc, test_fc = get_model(model_name)


plt.figure(figsize=(16,7))
if st.checkbox('Show forecast period ONLY'):
    plt.plot(test.groupby(pd.Grouper(freq='M')).sum(), label='Actual', linestyle = '--')
    #plt.plot(sarima_model.fittedvalues.groupby(pd.Grouper(freq='M')).sum(), label='Training')
    plt.plot(test_fc.groupby(pd.Grouper(freq='M')).sum(), label='Forecast', color= '#2ca02c')
else:
    plt.plot(df.groupby(pd.Grouper(freq='M')).sum(), label='Actual', linestyle = '--')
    plt.plot(train_fc.groupby(pd.Grouper(freq='M')).sum(), label='Training')
    #plt.plot(test.groupby(pd.Grouper(freq='M')).sum(), label='validation')
    plt.plot(test_fc.groupby(pd.Grouper(freq='M')).sum(), label='Forecast')

plt.legend(loc='upper left', fontsize=9)
plt.title(f"{model_name} Model Performance: Forecast vs Actual Orders", fontsize=18)
st.pyplot()

#------------------------------------------------
# Forecast plot comments
#------------------------------------------------
st.write('#### Dataset breakdown')
st.write('###### Training set period  : ',np.min(train_fc.index).strftime('%d/%m%/%Y'),
                                 ' - ', np.max(train_fc.index).strftime('%d/%m%/%Y'))
st.write('###### Validation set period: ', np.min(test_fc.index).strftime('%d/%m%/%Y'),
                                    ' - ', np.max(test_fc.index).strftime('%d/%m%/%Y'))

'''

'''

# Model Accuracy and Interpretability
train_mae = np.mean(np.abs(train_fc - train))
test_mae  = np.mean(np.abs(test_fc - test))
st.write('#### Model Accuracy using Mean Absolute Error (MAE)')
st.write('###### Model Training MAE: ',round(train_mae,3))
st.write('###### Model Forecast MAE: ',round(test_mae,3))
'''

'''
st.write('#### The metric above indicates that on average, the selected model is expected to be off by ', round(test_mae), 'orders when forecasting future sales.')




