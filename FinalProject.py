#!/usr/bin/env python
# coding: utf-8

# In[51]:


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from collections import defaultdict
from random import randint
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import plot_confusion_matrix

randSeed = 1936


# In[15]:


import warnings
warnings.filterwarnings("ignore")


# In[68]:


dataset = pd.read_csv('diabetes.csv', header = 'infer')
X = dataset.iloc[:, :-1].values
xdf = pd.DataFrame(X)
t = dataset.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(X, t, test_size=0.25, random_state=randSeed)
dataset.describe()


# In[13]:


import seaborn as sns
x_pd = pd.read_csv('diabetes.csv')
sns.pairplot(x_pd,kind='reg',diag_kind='hist')
correlations= x_pd.corr()
correlations


# In[78]:


####Scale X Data to be in Range [0,1]#####
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_pd_scale = sc.fit_transform(x_pd)
x_pd_df = pd.DataFrame(x_pd_scale)
x_pd_df = x_pd_df.rename(columns={0 : 'Pregnancies',1:'Glucose', 2:'BloodPressure', 3:'SkinThickness', 4:'Insulin', 5:'BMI', 6:'DiabetesPedigreeFunction', 7:'Age', 8: 'Outcome'})
x_pd_df.describe()

print(t.sum())


# In[12]:


####KNN####
neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(x_train, y_train)
neighTrain = neigh.predict(x_train)
neighPred = neigh.predict(x_test)
print("************3-NN Classifier******************")
print("******Confusion Matrix On Training Set************")
print(confusion_matrix(y_train,neighTrain))
print("******Classification Report On Training Set************")
print(classification_report(y_train,neighTrain))
print("******Confusion Matrix On Test Set************")
print(confusion_matrix(y_test,neighPred))
print("******Classification Report On Test Set************")
print(classification_report(y_test,neighPred))
RocCurveDisplay.from_estimator(neigh, x_test, y_test)
plt.title("3-NN ROC Curve")
plt.show()


# In[52]:


####Log Reg####
logReg = LogisticRegression()
logReg.fit(x_train, y_train)
logTrain = logReg.predict(x_train)
logPred = logReg.predict(x_test)
print("************Logistic Regression******************")
print("******Confusion Matrix On Training Set************")
print(confusion_matrix(y_train,logTrain))
print("******Classification Report On Training Set************")
print(classification_report(y_train,logTrain))
print("******Confusion Matrix On Test Set************")
print(confusion_matrix(y_test,logPred))
print("******Classification Report On Test Set************")
print(classification_report(y_test,logPred))
RocCurveDisplay.from_estimator(logReg, x_test, y_test)
plt.title("Logistic Regression ROC Curve")
plt.show()


# In[58]:


####Log Reg ####
logRegCV = LogisticRegressionCV(solver='sag', max_iter=10)
logRegCV.fit(x_train, y_train)
logCVTrain = logRegCV.predict(x_train)
logCVPred = logRegCV.predict(x_test)
print("************Logistic Regression******************")
print("******Confusion Matrix On Training Set************")
print(confusion_matrix(y_train,logCVTrain))
print("******Classification Report On Training Set************")
print(classification_report(y_train,logCVTrain))
print("******Confusion Matrix On Test Set************")
print(confusion_matrix(y_test,logCVPred))
print("******Classification Report On Test Set************")
print(classification_report(y_test,logCVPred))
RocCurveDisplay.from_estimator(logRegCV, x_test, y_test)
plt.title("Logistic Regression ROC Curve")
plt.show()


# In[42]:


####3 Layer Neural Net####
nNet = MLPClassifier(random_state=randSeed, hidden_layer_sizes=(8,8), max_iter=400)
nNet.fit(x_train, y_train)
nNetTrain = nNet.predict(x_train)
nNetPred = nNet.predict(x_test)
print("************Neural Net 3-Layer******************")
print("******Confusion Matrix On Training Set************")
print(confusion_matrix(y_train,nNetTrain))
print("******Classification Report On Training Set************")
print(classification_report(y_train,nNetTrain))
print("******Confusion Matrix On Test Set************")
print(confusion_matrix(y_test,nNetPred))
print("******Classification Report On Test Set************")
print(classification_report(y_test,nNetPred))
RocCurveDisplay.from_estimator(nNet, x_test, y_test)
plt.title("3-Layer (8,8,2) Neural Net ROC Curve")
plt.show()


# In[40]:


####5 Layer Neural Net####
nNetBig = MLPClassifier(random_state=randSeed, hidden_layer_sizes=(4,4,4,4), max_iter=400)
nNetBig.fit(x_train, y_train)
nNetBigTrain = nNetBig.predict(x_train)
nNetBigPred = nNetBig.predict(x_test)
print("************Neural Net 5-Layer******************")
print("******Confusion Matrix On Training Set************")
print(confusion_matrix(y_train,nNetBigTrain))
print("******Classification Report On Training Set************")
print(classification_report(y_train,nNetBigTrain))
print("******Confusion Matrix On Test Set************")
print(confusion_matrix(y_test,nNetBigPred))
print("******Classification Report On Test Set************")
print(classification_report(y_test,nNetBigPred))
RocCurveDisplay.from_estimator(nNetBig, x_test, y_test)
plt.title("5-Layer (4,4,4,4,2) Neural Net ROC Curve")
plt.show()


# In[76]:


####7 Layer Neural Net####
nNetMassive = MLPClassifier(random_state=randSeed, hidden_layer_sizes=(40,40,40,40,40,40), max_iter=1000)
nNetMassive.fit(x_train, y_train)
nNetMassiveTrain = nNetMassive.predict(x_train)
nNetMassivePred = nNetMassive.predict(x_test)
print("************Neural Net 7-Layer******************")
print("******Confusion Matrix On Training Set************")
print(confusion_matrix(y_train,nNetMassiveTrain))
print("******Classification Report On Training Set************")
print(classification_report(y_train,nNetMassiveTrain))
print("******Confusion Matrix On Test Set************")
print(confusion_matrix(y_test,nNetMassivePred))
print("******Classification Report On Test Set************")
print(classification_report(y_test,nNetMassivePred))
RocCurveDisplay.from_estimator(nNetMassive, x_test, y_test)
plt.title("7-Layer (40,40,40,40,40,40,2) Neural Net ROC Curve")
plt.show()


# In[75]:


####7 Layer Neural Net with Dropout####
nNetMassive = MLPClassifier(random_state=randSeed, hidden_layer_sizes=(40,40,40,40,40,40), max_iter=1000, early_stopping = True)
nNetMassive.fit(x_train, y_train)
nNetMassiveTrain = nNetMassive.predict(x_train)
nNetMassivePred = nNetMassive.predict(x_test)
print("************Neural Net 7-Layer with Dropout******************")
print("******Confusion Matrix On Training Set************")
print(confusion_matrix(y_train,nNetMassiveTrain))
print("******Classification Report On Training Set************")
print(classification_report(y_train,nNetMassiveTrain))
print("******Confusion Matrix On Test Set************")
print(confusion_matrix(y_test,nNetMassivePred))
print("******Classification Report On Test Set************")
print(classification_report(y_test,nNetMassivePred))
RocCurveDisplay.from_estimator(nNetMassive, x_test, y_test)
plt.title("7-Layer (40,40,40,40,40,40,2) Neural Net ROC Curve With Early Stopping")
plt.show()


# In[45]:


####Default Neural Net####
nNetDef = MLPClassifier(random_state=randSeed, max_iter=400)
nNetDef.fit(x_train, y_train)
nNetDefTrain = nNetDef.predict(x_train)
nNetDefPred = nNetDef.predict(x_test)
print("************Neural Net 5-Layer******************")
print("******Confusion Matrix On Training Set************")
print(confusion_matrix(y_train,nNetDefTrain))
print("******Classification Report On Training Set************")
print(classification_report(y_train,nNetDefTrain))
print("******Confusion Matrix On Test Set************")
print(confusion_matrix(y_test,nNetDefPred))
print("******Classification Report On Test Set************")
print(classification_report(y_test,nNetDefPred))
RocCurveDisplay.from_estimator(nNetDef, x_test, y_test)
plt.title("Default Neural Net ROC Curve")
plt.show()


# In[43]:


fig = plot_confusion_matrix(logReg, x_test, y_test, display_labels=nNet.classes_)
fig.figure_.suptitle("Confusion Matrix for Logistic Regression")
plt.show()
fig = plot_confusion_matrix(nNet, x_test, y_test, display_labels=nNet.classes_)
fig.figure_.suptitle("Confusion Matrix for 3-Layer Neural Net")
plt.show()
fig = plot_confusion_matrix(nNetBig, x_test, y_test, display_labels=nNet.classes_)
fig.figure_.suptitle("Confusion Matrix for 5-Layer Neural Net")
plt.show()
fig = plot_confusion_matrix(nNetMassive, x_test, y_test, display_labels=nNet.classes_)
fig.figure_.suptitle("Confusion Matrix for 7-Layer Neural Net")
plt.show()


# In[83]:


fig = plot_confusion_matrix(nNetBig, x_train, y_train, display_labels=nNet.classes_)
fig.figure_.suptitle("Confusion Matrix for 5-Layer Neural Net - Training Data")
plt.show()


# In[ ]:




