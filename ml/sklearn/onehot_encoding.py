'''
在实际的机器学习的应用任务中，特征有时候并不总是连续值，有可能是一些分类值，如性别可分为“male”和“female”。
在机器学习任务中，对于这样的特征，通常我们需要对其进行特征数字化，如下面的例子：

性别：["male"，"female"]
地区：["Europe"，"US"，"Asia"]
浏览器：["Firefox"，"Chrome"，"Safari"，"Internet Explorer"]

对于某一个样本，如["male"，"US"，"Internet Explorer"]，我们需要将这个分类值的特征数字化，
最直接的方法，我们可以采用序列化的方式：[0,1,3]。但是这样的特征处理并不能直接放入机器学习算法中。
对于上述的问题，性别的属性是二维的，同理，地区是三维的，浏览器则是思维的，
这样，我们可以采用One-Hot编码的方式对上述的样本“["male"，"US"，"Internet Explorer"]”编码，
“male”则对应着[1，0]，
同理“US”对应着[0，1，0]，
“Internet Explorer”对应着[0,0,0,1]。
则完整的特征数字化的结果为：[1,0,0,1,0,0,0,0,1]。这样导致的一个结果就是数据会变得非常的稀疏。
'''


import numpy as np
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1],[1, 0, 2]])

print("enc.n_values_ is:", enc.n_values_)
print("enc.feature_indices_ is:", enc.feature_indices_)
print(enc.transform([[0, 1, 1]]).toarray())

'''
这个代码很容易理解，简单解释一下没我一开始也没整明白。
首先由四个样本数据[0, 0, 3], [1, 1, 0], [0, 2, 1],[1, 0, 2]，共有三个属性特征，
也就是三列。比如第一列，有0,1两个属性值，第二列有0,1,2三个值.....

那么enc.n_values_就是每个属性列不同属性值的个数，所以分别是2,3,4

再看enc.feature_indices_是对enc.n_values_的一个累加。

再看[0, 1, 1]这个样本是如何转换为基于上面四个数据下的one-hot编码的。
第一列：0->10
第二列：1->010
第三列：1->0100

简单解释一下，在第三列有，0,1,2,3四个值，分别对应1000,0100,0010,0001.


REF:
https://blog.csdn.net/Mr_HHH/article/details/80006971
'''


