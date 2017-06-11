"""All training functions happen in here"""
import sys
sys.path.insert(0, "/data/")
import numpy as np
from data.review import is_regression
from reporting.scoring import scoring
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

def model_search(X, y, complexity='simple'):
	"""
	This function will decide what type of model to use
	Params:
		model class trained model object
		results dict trained model scoring information
		filename str filename being input
	Returns:
		True
	"""
	if is_regression(y):
		if complexity == 'simple':
			model = LinearRegression()
		else:
			model = xgboost.XGBRegressor()
	else:
		if complexity == 'simple':
			model = LogisticRegression()
		else:
			model = xgboost.XGBClassifier()
	return model

def train_and_validate(model, X, y):
	"""
	This function will train and cross validate your model
	Params:
		X np.array(matrix) input features
		y np.array(vector) target
		model class this is the model that will be trained and tested
	Returns:
		model class trained model object
		scoring() function returns scored dictionary
	"""
	ypred_class = np.zeros_like(y,dtype=float)                     # initialize holder array, make sure it is float
	ypred_prob = np.zeros_like(y,dtype=float)
	if is_regression(y):
		skf = KFold(n_splits=10, random_state=16)
		for train_index, test_index in skf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			model.fit(X_train,y_train)
			ypred_class[test_index] = model.predict(X_test)
	else: # must be classification
		skf = StratifiedKFold(n_splits=10, random_state=16)
		for train_index, test_index in skf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			model.fit(X_train,y_train)

			ypred_class[test_index] = model.predict(X_test)
			ypred_prob[test_index] = model.predict_proba(X_test)[:,1]
	return model, scoring(y,ypred_class,ypred_prob)
