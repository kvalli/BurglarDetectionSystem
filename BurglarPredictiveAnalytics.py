import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#Read Data
df = pd.read_csv("BurglarTrackingData.csv")
test_df = pd.read_csv("BurglarTrackingTestData.csv")

#Divide in to X and y to predict y based on other X data
X = df.drop(['Timestamp','burglar_detected'], axis=1)
y = df['burglar_detected']

#Read Test Data
#Separate in to test data and y actual to build model
test_data = test_df.drop(['Timestamp','burglar_detected'],axis =1)
y_test_actual = test_df['burglar_detected']

#Divding  data in to train and test data to build model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Use XGBoost classifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("XGBoost Model Accuracy",accuracy_score(y_test, y_pred)*100)
y_pred_new = model.predict(test_data)
print("XGBoost Test Accuracy",accuracy_score(y_test_actual, y_pred_new)*100)

#Apply random forest classifier to check
clf = RandomForestClassifier(max_depth=25, random_state=101)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Random Forest Model Accuracy",accuracy_score(y_test, y_pred)*100)
y_pred_new_rf = clf.predict(test_data)
print("Random Forest Test Accuracy",accuracy_score(y_test_actual, y_pred_new_rf)*100)

#Apply SVM classifier to check
svm_Clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_Clf.fit(X_train, y_train)

y_pred=svm_Clf.predict(X_test)
print("SVM Model Accuracy",accuracy_score(y_test, y_pred)*100)
y_pred_new_svm= model.predict(test_data)
print("SVM Test Accuracy",accuracy_score(y_test_actual, y_pred_new_svm)*100)