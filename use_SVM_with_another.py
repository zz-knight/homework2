
import numpy as np
from sklearn.svm import LinearSVC as LSVC
from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.model_selection import train_test_split


mnist = fetch_openml('mnist_784')
X=mnist["data"]
Y=mnist["target"]
X = X / 255.
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

lsvc=LSVC(penalty='l2',dual=True,loss='squared_hinge', tol=0.00001, C=0.02, multi_class='ovr',
          fit_intercept=True, intercept_scaling=1, class_weight=None,
          verbose=0, random_state=None, max_iter=2000)
lsvc.fit(X_train,Y_train)

train_accuracy=lsvc.score(X_train,Y_train)
test_accuracy=lsvc.score(X_test,Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))