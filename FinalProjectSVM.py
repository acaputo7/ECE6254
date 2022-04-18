import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.svm import SVC

#import dataset
ds = pd.read_csv('C:/Users/acaputo7/PycharmProjects/ECE6254/heart.csv')

#format dataset for model
ds_s=pd.get_dummies(ds)
X=ds_s.drop(['HeartDisease'],axis=1)
y=ds_s['HeartDisease']

#normalize data & split
scaler=StandardScaler()
X=scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#train SVC with no kernelization [gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma]
svm=SVC(class_weight='balanced')
svm.fit(X_train,y_train)

def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return np.mean(scores)


#predict
y_pred=svm.predict(X_test)



#evaluate accuracy
scores = []
cv_scores = []
scores.append(accuracy_score(y_pred,y_test))
cv_scores.append(evaluate_model(svm,X,y))
print(classification_report(y_pred,y_test))

