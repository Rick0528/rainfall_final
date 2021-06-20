import pandas as pd
import numpy as np

df = pd.read_csv('/content/rainfall_data.csv')

X = df.drop(['觀測時間(day)','測站最高氣壓時間(LST)','測站最低氣壓時間(LST)','最高氣溫時間(LST)','最低氣溫時間(LST)', '最小相對溼度時間(LST)',\
           '最大陣風風速時間(LST)','降水量(mm)','降水時數(hour)','最大十分鐘降水量(mm)', '最大十分鐘降水量起始時間(LST)', '最大六十分鐘降水量(mm)',\
       '最大六十分鐘降水量起始時間(LST)','日最高紫外線指數時間(LST)', ],axis=1)
X = X.iloc[1:-1]
Y = df['降水量(mm)'].iloc[2:]
Y = list(Y)
for i in range(len(Y)):
  if Y[i] == 'T':
    Y[i] = 1
Y = np.array(Y).astype(np.float)
Y = np.where(Y > 0, 1, 0)

def get_avg(df,idx):
  idxs = []
  before = (int)(idx/365)
  after = (int)(len(df)/365-before-1)
  sum = 0
  items = 0
  for i in range(before):
    tmp = df.iloc[idx-(365*(i+1))]
    if (tmp!='X' and tmp!='...' and tmp!='/'):
      items += 1
      sum += (float)(tmp)
  for i in range(after):
    tmp = df.iloc[idx+(365*(i+1))]
    if (tmp!='X' and tmp!='...'and tmp!='/'):
      items += 1
      sum += (float)(tmp)
  return sum/items

import unicodedata
from unicodedata import normalize
def clean_df(df):
  count = 0
  for n in df.columns:
    for i in df[n]:
      count += 1
      i = unicodedata.normalize("NFKD", i)
  print(count)

for n in X.columns:
  for i in range(len(X)):
    if X[n].iloc[i]=='X' or X[n].iloc[i]=='...' or X[n].iloc[i]=='/':
      X[n].iloc[i] = get_avg(X[n],i)

from sklearn import preprocessing
minmax = preprocessing.MinMaxScaler()
X_minmax = minmax.fit_transform(X)

X_minmax[0]

#RMSE for regression problems instead of roc_auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# ROC的分數
roc_values = []

# 計算分數
for feature in X.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X[feature].to_frame(), Y)
    y_scored = clf.predict_proba(X[feature].to_frame())
    roc_values.append(roc_auc_score(Y, y_scored[:, 1]))

# 建立Pandas Series 用於繪圖
roc_values = pd.Series(roc_values)
roc_values.index = X.columns

# 顯示結果
print(roc_values.sort_values(ascending=False))

np.save("./X", X_minmax)
np.save('./Y', Y)