from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
'''
交叉验证：

'''

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 4)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(knn.score(X_test, y_test))
print(y_test)
print(y_pred)

scores = cross_val_score(knn, X, y, cv=9, scoring='accuracy')
print("score is ", scores)
print("score means", scores.mean())


k_range = range(1, 30)
k_score = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    #k_score.append(scores.mean())

    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')
    k_score.append(loss.mean())

plt.plot(k_range, k_score)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
