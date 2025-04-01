import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Read the data
df=pd.read_csv('Palantir.csv')
df.index = pd.to_datetime(df['Date'])
df=df.drop(['Date'], axis='columns')

df['Open-Close']=df.Open-df.Close
df['High-Low']=df.High-df.Low
X=df[['Open-Close','High-Low']]
y=np.where(df['Close'].shift(-1)>df['Close'],1,-1)

split=0.8
split_index=int(split*len(df))

X_train=X[:split_index]
y_train=y[:split_index]
X_test=X[split_index:]
y_test=y[split_index:]

cls=SVC().fit(X_train,y_train)
df['Predicted_Signal']=cls.predict(X)
df['Return']=df['Close'].pct_change()
df['Strategy_Return']=df['Return']*df['Predicted_Signal'].shift(1)
df['Cumulative_Return']=np.cumsum(df['Return'])
df['Cumulative_Strategy_Return']=np.cumsum(df['Strategy_Return'])
df=df.dropna()

accuracy_train=accuracy_score(y_train,cls.predict(X_train))

accuracy_test=accuracy_score(y_test,cls.predict(X_test))

plt.figure(figsize=(10,5))
plt.title("SVM Stock Prediction for Palantir")
plt.xlabel('Date')
plt.plot(df['Cumulative_Return'], label='Cumulative Return')
plt.plot(df['Cumulative_Strategy_Return'], label='Cumulative Strategy Return')
plt.legend()
plt.show()

print('Stock Prediction for Palantir')
print('Strategic Return',df['Cumulative_Strategy_Return'].iloc[-1])
print('Return',df['Cumulative_Return'].iloc[-1])
print('Train Accuracy:',accuracy_train)

# Plot the data
"""""
plt.figure(figsize=(10,5))
plt.plot(df['Close'], label='Close Price history')
plt.title('Close Price history')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.legend(loc='upper left')
plt.show()


"""