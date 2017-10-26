
def classify(features_train, labels_train):

    from sklearn.naive_bayes import GaussianNB
### import the sklearn module for GaussianNB
    clf = GaussianNB()
### create classifier
    fit = clf.fit(features_train, labels_train)
### fit the classifier on the training features and labels
    return fit
### return the fit classifier


### your code goes here!






