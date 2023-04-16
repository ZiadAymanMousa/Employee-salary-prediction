#importing necessary libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from pprint import pprint
#######################################################################################################################

#READING CSV and defining unkowns
dtr = pd.read_csv('train.csv', na_values=[' ?', 'UNDEFINED'])
dte = pd.read_csv('test.csv', na_values=[' ?', 'UNDEFINED'])

#converting The data to DataFrame
DF_train = pd.DataFrame(dtr)
DF_test = pd.DataFrame(dte)

#Removing duplicates
DF_train.drop_duplicates()
DF_test.drop_duplicates()

print(DF_train.columns.str.strip())

#Dropping columns of relatively low correlation

DF_train = DF_train.drop(['race'], axis=1)
DF_test = DF_test.drop(['race'], axis=1)
# DF_test = DF_test.drop(['capital-gain'], axis=1)
# DF_train = DF_train.drop(['marital-status'], axis=1)
# DF_test = DF_test.drop(['marital-status'], axis=1)
# print(DF_train.isna().sum())

#Dropping nan values

DF_train.dropna(axis=0)
DF_test.dropna(axis=0)


#renaming columns of Train dataset to match the test ones
DF_train.rename(columns={'position': 'occupation'}, inplace=True)
DF_train.rename(columns={'work-class': 'workclass'}, inplace=True)
DF_train.rename(columns={'work-fnl': 'fnlwgt'}, inplace=True)

DF_train = DF_train.drop('fnlwgt',axis=1)
DF_test = DF_test.drop('fnlwgt',axis=1 )
DF_train = DF_train.drop('relationship', axis=1)
DF_test = DF_test.drop('relationship' , axis=1)
#array of columns to encode

col1 = ['workclass', 'education', 'marital-status', 'occupation',  'sex', 'native-country', 'salary']
col2 = ['workclass', 'education', 'marital-status',  'occupation', 'sex', 'native-country']

#for loop for using Label encoding for each of the columns values

def feature_encoder(d, column):
    for c in column:
        lbl = LabelEncoder()
        lbl.fit(list(d[c].values))
        d[c] = lbl.transform(list(d[c].values))
    return d

#applying encoder function on both arrays


feature_encoder(DF_train, col1)
feature_encoder(DF_test, col2)

for column in DF_train.columns:
 lower_bound  = DF_train[column].mean() - 3 * DF_train[column].std()
 upper_bound = DF_train[column].mean() + 3 * DF_train[column].std()
 DF_train[column] = DF_train[column].clip(lower_bound, upper_bound)

#scaling data to minimize difference between big and relatively small values

ww = DF_train.columns
sc = MaxAbsScaler()
DF_train = sc.fit_transform(DF_train)
DF_train = pd.DataFrame(DF_train, columns=ww)

# norm = MinMaxScaler().fit(DF_train)
#
# DF_train = norm.transform(DF_train)
#
# DF_test = norm.transform(DF_test)

#splitting train dataset to train and test
# print(DF_train.columns)
# print(DF_test.columns)
X = DF_train.iloc[:, 0:11]
Y = DF_train['salary']
K = DF_test
X_train, X_vtest, Y_train, Y_vtest = train_test_split(X, Y, random_state=10, test_size=0.2, shuffle=True)

#Trying out different models and observing their accuracy output 

#1 Logistic Regression model

# logreg = linear_model.LogisticRegression()
# logreg.fit(X_train, Y_train)
# pred = logreg.predict(X_vtest)
#
# DTF = pd.DataFrame(pred, columns=['salary'])
#
# sns.countplot(x='salary', data=DTF)
# plt.show()
#
#
# DTF.replace(to_replace=1.0, value=' >50K', inplace=True)
# DTF.replace(to_replace=0, value=' <=50K', inplace=True)
#
# accur = accuracy_score(Y_vtest, pred)
# accuracy_perc = 100*accur
# print("Accuracy percentage is ", accuracy_perc, "%")
# # DTF.to_csv('last.csv')








##############################################################################################################

#2 GaussianNB Model

#
# gnb = GaussianNB()
# gnb.fit(X_train, Y_train)
# pred = gnb.predict(K)
#
# DTF = pd.DataFrame(pred, columns=['salary'])
#
#
# sns.countplot(x='salary', data=DTF)
# plt.show()
#
#
# DTF.replace(to_replace=1.0, value=' >50K', inplace=True)
# DTF.replace(to_replace=0, value=' <=50K', inplace=True)
#
# accur = accuracy_score(Y_vtest, pred)
# accuracy_perc = 100*accur
# print("Accuracy percentage is ", accuracy_perc, "%")
# DTF.to_csv('late5.csv')

##########################################################################

#3 SVC

# svc = SVC()
# svc.fit(X_train, Y_train)
# pred = svc.predict(X_vtest)
# DTF = pd.DataFrame(pred, columns=['salary'])
#
#
# sns.countplot(x='salary', data=DTF)
# plt.show()
#
#
# DTF.replace(to_replace=1.0, value=' >50K', inplace=True)
# DTF.replace(to_replace=0, value=' <=50K', inplace=True)
#
# accur = accuracy_score(Y_vtest, pred)
# accuracy_perc = 100*accur
# print("Accuracy percentage is ", accuracy_perc, "%")
# # DTF.to_csv('last2.csv')

###################################################################


#4 KNeighbors Model
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# pred = knn.predict(X_vtest)
# DTF = pd.DataFrame(pred, columns=['salary'])
#
#
# sns.countplot(x='salary', data=DTF)
# plt.show()
#
#
# DTF.replace(to_replace=1.0, value=' >50K', inplace=True)
# DTF.replace(to_replace=0, value=' <=50K', inplace=True)
#
# accur = accuracy_score(Y_vtest, pred)
# accuracy_perc = 100*accur
# print("Accuracy percentage is ", accuracy_perc, "%")
# DTF.to_csv('last4.csv')

######################################################################

#5 Random Forest

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
pred = rfc.predict(X_vtest)
DTF = pd.DataFrame(pred, columns=['salary'])
# print(rfc.get_params())
{'bootstrap': True,
 'criterion': 'mse',
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 10,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': 42,
 'verbose': 0,
 'warm_start': False}

print('Parameters currently in use:\n')
print(rfc.get_params())
sns.countplot(x='salary', data=DTF)
plt.show()


DTF.replace(to_replace=1.0, value=' >50K', inplace=True)
DTF.replace(to_replace=0, value=' <=50K', inplace=True)

accur = accuracy_score(Y_vtest, pred)
accuracy_perc = 100*accur
print("Accuracy percentage is ", accuracy_perc, "%")
# DTF.to_csv('late21.csv')







#junk Lines

#print(j_scaled)
#print(DF_train['position'].value_counts())
#print(DF_train.isna().sum())
#total_num_rows_dtr = len(dtr)
#total_num_rows_dte = len(dte)
#dtr_clean = dtr[~dataframe_train]
#dte_clean = dte[~dataframe_test]
#dte_cols_nan = dte.columns[mask2]
#dataframe_train = dataframe_train.isnull().any(axis=0)
#dataframe_test = dataframe_test.isnull().any(axis=0)
#um_rows_nan_train = dataframe_train.sum()
#num_rows_nan_test = dataframe_test.sum()

#scaler = Normalizer()
#scaler.fit(DF_test)
#DF_test = scaler.transform(DF_test)
#DF_test = pd.DataFrame(DF_test, columns=DF_test.columns)
#print(j_scaled[5:10])
#col3 = ['work-fnl', 'capital-gain', 'capital-loss']
#col4 = ['fnlwgt', 'capital-gain', 'capital-loss']
#print(DF_train['salary'].value_counts())
#DF_train = DF_train.dropna(axis=0, how='any')
#DF_test = DF_test.dropna(axis=0, how='any')
#DF_train.fillna(method='ffill', inplace=True)
#DF_test.fillna(method='ffill', inplace=True)