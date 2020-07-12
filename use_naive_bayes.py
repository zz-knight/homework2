
import numpy as np
from sklearn.naive_bayes import BernoulliNB as BL
from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.model_selection import train_test_split


mnist = fetch_openml('mnist_784')
X=mnist["data"]
Y=mnist["target"]
X = X / 255.
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

bl=BL()
bl.fit(X_train,Y_train)

train_accuracy=bl.score(X_train,Y_train)
test_accuracy=bl.score(X_test,Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))