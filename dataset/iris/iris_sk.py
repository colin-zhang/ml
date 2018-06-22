#!/usr/bin/env python3

from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn

iris = load_iris()
#print(iris)

knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)
predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print(predict)
print(iris.target_names[predict])
