import pandas as pd

df_iris= pd.read_csv("D:\Hero_ML\iris.csv")
# X, y = df_iris
df_iris.head()
df_iris.describe()

#normalization
from sklearn.preprocessing import MinMaxScaler
norm =MinMaxScaler()
df_norm=norm.fit_transform(df_iris.iloc[:,:4])

#converting the normalized data to data frame
df_norm=pd.DataFrame(df_norm)
df_norm.describe()

X=df_norm
y=df_iris['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=.2)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model=rf.fit(X_train, y_train)

#predcition on test data

pred_test=model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, pred_test)
print(classification_report(y_test, pred_test))


# dumping model into pickle file
import pickle

pickle.dump(model,open("model.pkl", "wb"))

import os
os.getcwd()


