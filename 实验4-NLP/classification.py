import sklearn
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import pandas as pd

def classification(processed_df):
	# split into train and test sets
	kf = KFold(n_splits = 10, shuffle = True, random_state = 2)
	split_result = next(kf.split(processed_df), None)
	train = processed_df.iloc[split_result[0]]
	test = processed_df.iloc[split_result[1]]
	
	# TF-IDF
	X_train_tf, X_test_tf = TF_IDF(train, test)

	# classification
	# Your code here
    # Answer begin


	vc = processed_df['c_name'].value_counts()
	vc = pd.Series([i for i in range(len(vc))], index=vc.index)
	y_train = np.array([vc[train['c_name'][i]] for i in train.index])
	y_test = np.array([vc[test['c_name'][i]] for i in test.index])


	clf = MultinomialNB()
	clf.fit(X_train_tf, y_train)

	y_predict = clf.predict(X_test_tf)

	print("test['c_name'] type is", type(test['c_name']))
	print(" y_predict type is",  y_predict)
	results = metrics.classification_report(test['c_name'], y_predict, target_names=vc.index)
	# ??,target_names = vc.index

	# Answer end
	return results

def TF_IDF(train, test):
	# Your code here
    # Answer begin

	vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	X_train_tf = transformer.fit_transform(vectorizer.fit_transform(train['business_scope'].values)).toarray()
	X_test_tf = transformer.transform(vectorizer.transform(test['business_scope'].values)).toarray()

	# Answer end
	return X_train_tf, X_test_tf
