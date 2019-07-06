
# part -1  data preprocessing

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading data set
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13].values
y = data.iloc[:,-1:].values


# Encoding categorial data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder1=LabelEncoder()
label_encoder2=LabelEncoder()
X[:,1]=label_encoder1.fit_transform(X[:,1])
X[:,2]=label_encoder2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X =onehotencoder.fit_transform(X).toarray()
X =X[:,1:]

# training and testing data splittting
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y =train_test_split(X ,y ,test_size=0.20 ,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
train_X = sc_x.fit_transform(train_X)
test_X = sc_x.transform(test_X)


# part -2 Now make an ANN network

#import keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense 

#initialising ANN
classifier=Sequential()

# adding input layer that is input layer                              #unit means output nodes and input sim is input nodes
classifier.add(Dense(units=6, activation='relu',input_dim=11 ))
#adding secoung hidden layer
classifier.add(Dense(units=6, activation='relu' ))
#adding output result
classifier.add(Dense(units=1,activation='sigmoid'))

#compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#fitting ANN to training set
classifier.fit(train_X,train_Y, epochs=40, batch_size=30)

# part -3 making prediction and result

#predictions of result
y_pred =classifier.predict(test_X)
y_pred=(y_pred>0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm =confusion_matrix(test_Y ,y_pred)
accuracy_score(test_Y,y_pred)*100