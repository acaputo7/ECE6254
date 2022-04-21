import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import plotly.express as px




#import dataset
DATA = pd.read_csv('C:/Users/acaputo7/PycharmProjects/ECE6254/heart.csv')

#split categorical features and numerical features
numerical = DATA.drop(['HeartDisease'], axis=1).select_dtypes('number').columns

categorical = DATA.select_dtypes('object').columns

# print(f'Numerical Columns:  {DATA[numerical].columns}')
# print('\n')
# print(f'Categorical Columns: {DATA[categorical].columns}')

# #plot Pearson heatmap
# plt.figure(1, figsize=(12, 8))
# sns.heatmap(DATA.corr(), annot=True)
# plt.xticks(rotation=45)
# # plt.savefig('correlation.png')
#
# #plot distribution of numerical variables with presence of heard disease
#
# plt.rcParams.update({'font.size':14})
# sns.pairplot(DATA, hue="HeartDisease", corner=True)
# # plt.savefig('pairplot.png')

#plot distributions of categorical variables
#
# fig5 = px.histogram(DATA,
#                  x="Sex",
#                  color="HeartDisease",
#                  hover_data=DATA.columns,
#                 )
# fig5.show()
#
# fig1 = px.histogram(DATA,
#                  x="ChestPainType",
#                  color="HeartDisease",
#                  hover_data=DATA.columns,
#                 )
# fig1.show()
#
# fig2 = px.histogram(DATA,
#                  x="RestingECG",
#                  color="HeartDisease",
#                  hover_data=DATA.columns,
#                 )
# fig2.show()
#
# fig3 = px.histogram(DATA,
#                  x="ExerciseAngina",
#                  color="HeartDisease",
#                  hover_data=DATA.columns,
#                 )
# fig3.show()
#
# fig4 = px.histogram(DATA,
#                  x="ST_Slope",
#                  color="HeartDisease",
#                  hover_data=DATA.columns,
#                 )
# fig4.show()







#format dataset for model by splitting class from feature list
DATA_s=pd.get_dummies(DATA)
X = DATA_s.drop(['HeartDisease'], axis=1)
y = DATA_s['HeartDisease']


#normalize data & split
scaler = StandardScaler()
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=25)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#train SVC  [gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma]
svm=SVC(kernel='rbf', class_weight='balanced')
clf = svm.fit(X_train, y_train)
print(svm.get_params())
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    scores = cross_val_score(model, scaler.fit_transform(X), y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return np.mean(scores)


#predict
y_pred = svm.predict(X_test)

#confusion matrix

# disp = ConfusionMatrixDisplay(confusion_matrix(y_true=y_test,y_pred=y_pred))
# disp.plot()
# plt.savefig('Confusionmatrix.png')

#plot ROC curve
# disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
# disp.plot()
# plt.savefig('ROC_CurveSVM.png')
#evaluate accuracy
score = []
cv_scores = []
score.append(accuracy_score(y_pred=y_pred, y_true=y_test))
cv_scores.append(evaluate_model(svm, X, y))
print(classification_report(y_pred=y_pred, y_true=y_test))
print('Cross validation mean score: ' + str(cv_scores))
print('Accuracy score: ' + str(score))
