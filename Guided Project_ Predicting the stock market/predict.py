import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime


df= pd.read_csv('sphist.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date', ascending=True)
#rint(df[:2])

df['day_5'] = 0
df['day_30'] = 0
df['day_365'] = 0
#rint(df[:2])

df['day_5'] = df['Close'].rolling(5).mean()
df['day_5'] = df['day_5'].shift()
df['day_30'] = df['Close'].rolling(30).mean()
df['day_30'] = df['day_30'].shift()
df['day_365'] = df['Close'].rolling(365).mean()
df['day_365'] = df['day_365'].shift()

df = df.dropna(axis=0)
train = df[df['Date'] < datetime(year=2013,month=1,day=1)]
test = df[df['Date'] >= datetime(year=2013,month=1,day=1)]

model = LinearRegression()
model.fit(train[['day_5']],train['Close'])
predicions = model.predict(test[['day_5']])
mae = mean_absolute_error(predicions, test['Close'])
print('day_5: ',mae)

model.fit(train[['day_30']],train['Close'])
predicions = model.predict(test[['day_30']])
mae = mean_absolute_error(predicions, test['Close'])
print('day_30: ',mae)

model.fit(train[['day_365']],train['Close'])
predicions = model.predict(test[['day_365']])
mae = mean_absolute_error(predicions, test['Close'])
print('day_365: ',mae)

model.fit(train[['day_5','day_30']],train['Close'])
predicions = model.predict(test[['day_5','day_30']])
mae = mean_absolute_error(predicions, test['Close'])
print('day_5_30: ',mae)

model.fit(train[['day_5','day_365']],train['Close'])
predicions = model.predict(test[['day_5','day_365']])
mae = mean_absolute_error(predicions, test['Close'])
print('day_5_365: ',mae)

model.fit(train[['day_30','day_365']],train['Close'])
predicions = model.predict(test[['day_30','day_365']])
mae = mean_absolute_error(predicions, test['Close'])
print('day_30_365: ',mae)

model.fit(train[['day_5','day_30','day_365']],train['Close'])
predicions=model.predict(test[['day_5','day_30','day_365']])
mae = mean_absolute_error(predicions, test['Close'])
print('day_5_30_365: ',mae)