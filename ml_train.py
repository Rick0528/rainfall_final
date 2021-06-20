import numpy as np
import pandas as pd

x = np.load("X.npy")
y = np.load("Y.npy")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 100)

#tree
from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)

dt.fit(X_train, y_train)
treeScore = dt.score(X_test, y_test)
print("\ntree : ", treeScore)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
print("\nLogisticRegression : ", clf.score(X_test, y_test))

#SVM
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
svc = SVC(C=1, kernel='linear')    #試了10 1 0.1 0.01發現跑出來的score沒有太大的差異
svc.fit(X_train, y_train)
print("\nSVC score : ", svc.score(X_test, y_test))
svcCV = cross_val_score(svc, x, y, scoring='accuracy', cv=10)
print("Cross Validation(SVC) : ")
print(svcCV)
print("Mean : ", svcCV.mean())

#forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500, max_depth=5)

rfc.fit(X_train, y_train)
Score = rfc.score(X_test, y_test)
print("\nRandomForestClassifier : ", Score)