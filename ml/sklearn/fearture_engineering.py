from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.datasets import load_iris

'''
参考：
http://www.cnblogs.com/jasonfreak/p/5448385.html
'''

iris = load_iris()
iris_data = iris.data
iris_target = iris.target

print(iris_data, "\n", iris_target)

one_hot_enconding_data = OneHotEncoder().fit_transform(iris.target.reshape((-1, 1)))

print(one_hot_enconding_data)


nor_data = Normalizer().fit_transform(iris.data)
print(nor_data)