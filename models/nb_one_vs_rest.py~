#!/usr/bin/env python

## This file implements One-vs-rest classifier strategy for multi-label classification with Logistic Regression as the base model

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# Reading the files..
print('reading started')
X_train, y_train = load_svmlight_file("../Data/wise2014-train.libsvm", dtype=np.float64, multilabel=True)
X_test, y_test = load_svmlight_file("../Data/wise2014-test.libsvm", dtype=np.float64, multilabel=True)
print('reading finished')

# Binarize labels
lb = MultiLabelBinarizer()
y_train = lb.fit_transform(y_train)


# fit and predict for each class - iterative version used because of low memory of hardware platform
y_final = []
for i in range(y_train.shape[1]):
	print('i : '+str(i))

	clf = LogisticRegression()

	print('fitting started')
	clf.fit(X_train, y_train[:,i])
	print('fitting finished')

	print('predict started')
	pred_y = clf.predict(X_test)
	print('predict finshed')

	y_final.append(pred_y)

y_final = np.array(y_final).T

# Writing the output to a file
out_file = open("pred.csv","w")
out_file.write("ArticleId,Labels\n")
id = 64858
for i in xrange(y_final.shape[0]):
	label = list(lb.classes_[np.where(y_final[i,:]==1)[0]].astype("int"))
	label = " ".join(map(str,label))
	## If the label is empty, populate the most frequent label
	if label == "":
		label = "103"
	out_file.write(str(id+i)+","+label+"\n")
out_file.close()
