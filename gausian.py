import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
# tipo de classificador
clf.fit(X, Y)
# treinamento x[classe] y[labels]

print(clf.predict([[2, 1]]))
# resultado a qual classe pertence o objeto [2,1]

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

# print(clf_pf.predict([[-0.8, -1]]))