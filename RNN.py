# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:37:20 2020

@author: Mucahit Kartal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('spambase.data', names = ["make", "adress", "all", "3d", "our", "over", "remove", "internet", "order", 
                                                "mail", "receive", "will", "people", "report", "addresses", "free", "business" ,"email",
                                                "you", "credit", "your", "font", "000", "money", "hp", "hpl", "george", "650", 
                                                "lab", "labs", "telnet", "857", "data", "415", "85", "technology", "1999", 
                                                "parts", "pm", "direct", "cs", "meeting", "original", "project", "re", "edu", 
                                                "table", "conference", ";", "(", "[", "!", "$", "#", "length_average", "length_longest",
                                                "length_total" ,"spam"])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
x = veriler.iloc[:,:57].values#48-57
y = veriler.iloc[:,-1:].values

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split  

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

print(X_train.shape)
print(X_train.shape[1])

#keras, RNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, RNN, GRU

#LSTM kullanımı için reshape işlemi
X_train = X_train.reshape(-1, 1, 57)
X_test  = X_test.reshape(-1, 1, 57)
y_train = y_train.reshape(-1, 1, 1)

model = Sequential()

#model.add(SimpleRNN(128, activation = 'relu', return_sequences = True, input_shape=(X_train.shape[1:])))
#model.add(GRU(128, return_sequences = True,recurrent_activation = sigmoid, recurrent_dropout = 0, unroll=0, use_bias=1, reset_after=1, activation = 'tanh', input_shape=(X_train.shape[1:])))
model.add(LSTM(128, return_sequences = True, activation = 'tanh', input_shape=(X_train.shape[1:])))
model.add(Dropout(0.2))
#model.add(SimpleRNN(128, activation = 'relu'))
model.add(LSTM(128, activation = 'tanh'))
model.add(Dropout(0.2))

model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.2))

#model.add(LSTM(units = 50, activation = 'relu'))
#model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid'))


model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 50, validation_data=(X_test,y_test))

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)

#Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)


#AUC
y_proba = model.predict_proba(X_test)

from sklearn.metrics import roc_curve, auc, roc_auc_score

#pos_label=0
fpr, tpr, thold = roc_curve(y_test, y_proba[:,0], pos_label=1)

roc_auc= auc(fpr,tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RNN(SimpleRNN)-Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




