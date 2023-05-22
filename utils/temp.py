# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:03:32 2023

@author: ocean
"""

#%% 1.载入使用的包
import pandas as pd
import matplotlib.pyplot as plt

import re
import paddle
from paddlets.datasets.tsdataset import TSDataset
from paddlets.transform import TimeFeatureGenerator, StandardScaler
from paddlets.models.forecasting import LSTNetRegressor
from paddlets.metrics import MAE
import warnings
warnings.filterwarnings('ignore')
#from paddlets.automl.autots import AutoTS
#%% 2. 读取文件并定义提取站信息
data = pd.read_csv('data/客流量数据.csv')
df = data.copy()
def extract_station(x):    
    station = re.compile("([A-Z])站").findall(x)
    if len(station) > 0:
        return station[0]
#%% 3.时间格式修改
df["时间区段"] = df["时间区段"].apply(lambda x:x.split("-")[0])
df["站点"] = df["站点"].apply(extract_station)
df["时间"] = df["时间"].apply(lambda x:str(x)).str.cat(df['时间区段'],sep=" ")
df["时间"] = pd.to_datetime(df["时间"])
df = df.drop("时间区段",axis=1)
df.columns =["Time", "Station", "InNum", "OutNum"] 
print(df.head())
#%% 4.选择一个站，用于测试效果
station = "C" # 站点
dataset_df = df[df['Station']==station]
print(dataset_df.head())
#%% 5.将dataframe格式的数据读取为ts格式
dataset_df = TSDataset.load_from_dataframe(
    dataset_df,
    time_col='Time', #时间序列
    target_cols=['InNum','OutNum'], #预测目标
    freq='15min',
    fill_missing_dates=True,
    fillna_method='zero'
)
print(dataset_df.label)

dataset_df.plot()
plt.show()
dataset_df.summary()

#%% 6.切分数据集 并标准化
dataset_train, dataset_val_test = dataset_df.split("2023-02-05 23:45:00")
dataset_val, dataset_test = dataset_val_test.split("2023-03-29 23:45:00")

scaler = StandardScaler()
scaler.fit(dataset_train)
dataset_train_scaled = scaler.transform(dataset_train)
dataset_val_test_scaled = scaler.transform(dataset_val_test)
dataset_val_scaled = scaler.transform(dataset_val)
dataset_test_scaled = scaler.transform(dataset_test)
#%% 7. 定义模型
paddle.seed(2023)
model = LSTNetRegressor(
    in_chunk_len = 4*24,
    out_chunk_len = 4*24,
    max_epochs = 200,
    patience = 20,
    eval_metrics =["mae"],
    optimizer_params= dict(learning_rate=3e-3)
)
#%% 8.模型拟合
model.fit(dataset_train_scaled, dataset_val_scaled)
model._history
