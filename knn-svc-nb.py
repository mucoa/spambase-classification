# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:42:02 2020

@author: DELL
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
x = veriler.iloc[:,:57].values
y = veriler.iloc[:,-1:].values

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split  

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state = 0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)      


#knn sınıflandrıması
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski')
knn.fit(X_train,y_train)

knn_y_pred = knn.predict(X_test)

#svm
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', probability=1)
svc.fit(X_train,y_train)

svc_y_pred = svc.predict(X_test)

#naive bayes
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
bNB = MultinomialNB()
bNB.fit(abs(X_train), y_train)

bNB_y_pred = bNB.predict(X_test)


#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,bNB_y_pred)    
print(cm)


#ROC-AUC grafik çıkarımı
knn_y_proba = knn.predict_proba(X_test)
svc_y_proba = svc.predict_proba(X_test)
bNB_y_proba = bNB.predict_proba(X_test)

print(y_test)
print(knn_y_proba[:,0])

from sklearn.metrics import roc_curve, auc, roc_auc_score

#pos_label=0
fpr, tpr, thold = roc_curve(y_test, bNB_y_proba[:,0], pos_label=0)

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
plt.title('NB-Multinominal-Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()