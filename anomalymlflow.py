#!/usr/bin/env python
# coding: utf-8
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
from numpy import log
#import plotly_express as px
from datetime import datetime
from datetime import timedelta, date
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mlflow
import mlflow.sklearn
import os
import warnings
import sys
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
# mlflow.set_tracking_uri('http://127.0.0.1:5000')  # set up connection
# mlflow.set_experiment('test-experiment')          # set the experiment
# mlflow.pytorch.autolog()
#implement model
#------------------------------------------------
class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length):

        super(BayesianLSTM, self).__init__()

        self.hidden_size_1 = 128
        self.hidden_size_2 = 32
        self.n_layers = 1 # number of (stacked) LSTM layers

        self.lstm1 = nn.LSTM(n_features, 
                             self.hidden_size_1, 
                             num_layers=1,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=1,
                             batch_first=True)
        
        self.dense = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=0.5, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=0.5, training=True)
        output = self.dense(state[0].squeeze(0))
        
        return output
        
    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        return hidden_state, cell_state
    
    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_2))
        return hidden_state, cell_state
    
    def loss(self, pred, truth):
        loss = self.log(loss_fn(pred, truth)) 
        return loss

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()
#------------------------------------------------
# preparing data for prediction
def clean_df(dff):
    #Clean Data 
    #dff = pd.read_csv("webs.csv", parse_dates=[['date', 'time']])
    df1 =dff.dropna(axis=0)
    df1 = df1.rename(columns={"Time":"time"})
    df1 = df1.rename(columns={"Count":"count"})
    df1['time'] = pd.to_datetime(df1['time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    df1['time'].astype('datetime64[ns]')
    df=df1.dropna(axis=0)
    df = df[['time', 'count']]
    df['time'] = pd.to_datetime(df['time'])
    #print rage time
    #df['time'].min(), df['time'].max()
    time5= df['time']
    dd = {'date':time5}
    dataa = pd.DataFrame(dd)
    dataa['count']=df['count']
    
    #group time to 5 mins.
    dataa.date = dataa.date.apply(str)
    dataa.date = dataa.date.apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    data2=dataa.groupby(pd.Grouper(key='date', freq='5min'))['count'].count()  
    #---------------------------------------------------------------------
    
    # Reset index
    b= data2.reset_index(drop=False)
    resample_df = b.rename(columns={"date":"date","count": "counts"})
    resample_df['date'] = pd.to_datetime(resample_df['date'])
    resample_df['month'] = resample_df['date'].dt.month.astype(int)
    resample_df['day_of_month'] = resample_df['date'].dt.day.astype(int)
    # day_of_week=0 corresponds to Monday 
    # extract Date
    resample_df['day_of_week'] = resample_df['date'].dt.dayofweek.astype(int)
    resample_df['hour_of_day'] = resample_df['date'].dt.hour.astype(int)
    # Resample by mean and change scale with log
    selected_columns = ['date','day_of_week','hour_of_day','day_of_month','counts']
    resample_df = resample_df[selected_columns]
    resample_df = resample_df.set_index('date').resample('5min').mean()
    resample_df['date'] = resample_df.index
    resample_df['log_counts'] = np.log(resample_df['counts']+1, where=resample_df['counts'] > 0.0000000001)
    datetime_columns = ['date', 'day_of_week', 'hour_of_day','day_of_month']
    target_column = 'log_counts'
    feature_columns = datetime_columns + ['log_counts']
    #------------------------------------------------------------------------------------------------
    # finish cleaning
    resample_df = resample_df[feature_columns]
    return resample_df 
#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
#create sliding window  for time series model
def create_sliding_window(data, sequence_length, stride=1):
    X_list, y_list = [], []
    for i in range(len(data)):
      if (i + sequence_length) < len(data):
        X_list.append(data.iloc[i:i+sequence_length:stride, :].values)
        y_list.append(data.iloc[i+sequence_length, -1])
    return np.array(X_list), np.array(y_list)

# inverse data after transform scaler
def inverse_transform(y):
    return target_scaler.inverse_transform(y.reshape(-1, 1))


## import testing set

# df = pd.read_csv("5mcount.csv")
# df.Time = pd.to_datetime(df.Time).dt.tz_localize(None)
# resample_test = clean_df(df)

# f={
#     "Time": "2020-11-17 14:00:00",
#     "Count": "2100"}
# df1 = pd.json_normalize(f)
# df = df1[['Time', 'Count']]
# resample_test = clean_df(df)

# defind data type before get data from API
class count(BaseModel):
  Time: str
  Count: float

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # 1.import taining data for training data cleaning and import testiing data

    # df1 = pd.json_normalize(f)
    # df = df1[['Time', 'Count']]
    # resample_test = clean_df(df)
    dff = pd.read_csv("rf.csv") 
    resample_df = clean_df(dff)
    train = resample_df.iloc[:100000,:]
    test = resample_df.iloc[-100000:,:] #resample_test
    resample_df = train.append(test, ignore_index=True)
    n_train = len(train)-10
    n_test = len(test)

    # 2.data cleaning and extracted feature time from data
    features = ['day_of_week', 'hour_of_day', 'day_of_month','log_counts']
    feature_array = resample_df[features].values
    # Fit Scaler only on Training features
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_array[:n_train])
    # Fit Scaler only on Training target values
    target_scaler = MinMaxScaler()
    target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))
    # Transfom on both Training and Test data
    scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                                columns=features)
    sequence_length = 10
    X, y = create_sliding_window(scaled_array, 
                                sequence_length)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:]
    y_test = y[n_train:]

    # 3.import model and parameter for model
    n_features = scaled_array.shape[-1]
    sequence_length = 10
    output_length = 1

    bayesian_lstm = BayesianLSTM(n_features=n_features,
                             output_length=output_length)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=0.01)
    #alpha = float(sys.argv[1])
    batch_size = 128
    n_epochs = int(sys.argv[1]) 
    with mlflow.start_run():
    # Train
        bayesian_lstm.train()
        for e in range(1, n_epochs+1):
            accuracies = 0
            for b in range(0,len(X_train), batch_size):
                features = X_train[b:b+batch_size,:,:]
                target = y_train[b:b+batch_size]    

                X_batch = torch.tensor(features,dtype=torch.float32)    
                y_batch = torch.tensor(target,dtype=torch.float32)

                output = bayesian_lstm(X_batch) 
                loss = criterion(output.view(-1), y_batch)  

                loss.backward()
                optimizer.step()        
                optimizer.zero_grad()
                matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(output, y_batch)]
                acc = matches.count(True)/len(matches)
                #print("Test Accuracy:", round(acc, 3))

            if e % 10 == 0:
                
                print('epoch: ', e, 'loss: ', loss.item(), 'acc: ', acc)
            print (loss)
    
        # make predictions
        y_test_pred = bayesian_lstm.predict(X_test)

        # invert predictions
        y_train_pred = bayesian_lstm.predict(X_train)
        y_train_pred = inverse_transform(y_train_pred)
        y_train = inverse_transform(y_train)
        y_test_pred = inverse_transform(y_test_pred)
        y_test = inverse_transform(y_test)

        # calculate root mean squared error
        trainScoreR = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
        print('Train Score: %.2f RMSE' % (trainScoreR))
        testScoreR = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
        print('Test Score: %.2f RMSE' % (testScoreR))


        # calculate root mean absolute error
        trainScoreA = math.sqrt(mean_absolute_error(y_train[:,0], y_train_pred[:,0]))
        print('Train Score: %.2f MAE' % (trainScoreA))
        testScoreA = math.sqrt(mean_absolute_error(y_test[:,0], y_test_pred[:,0]))
        print('Test Score: %.2f MAE' % (testScoreA))


        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_metric("testScoreR", testScoreR)
        mlflow.log_metric("testScoreA", testScoreA)
    # bayesian_lstm.state_dict()
    # isinstance(bayesian_lstm, nn.Module)
    # for name, child in bayesian_lstm.named_children():
    #     print('name: ', name)
    #     print('isinstance({}, nn.Module): '.format(name), isinstance(child, nn.Module))
    #     print('=====')
    
    # torch.save(bayesian_lstm.state_dict(), 'weights_only_log.pth')

    