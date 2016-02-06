# -*- coding: utf-8 -*-
import pickle

from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import *
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plot.plots import *


class BaseModel(object):
    def rmse(self, y, p):
        err = abs(p-y)
        total_error = np.dot(err,err)
        rmse = np.sqrt(total_error/len(p))
        return rmse

    def vif(self, X, features):
        model = LinearRegression()
        for column in features:
            Xk = X.drop(column, 1)
            model.fit(Xk,X[column])
            R2 = model.score(Xk,X[column])
            vif_score = 1.0 / (1.0 - R2)
            alert = '' if np.sqrt(vif_score) < 2 else '*'
            print '%20s VIF: %f' % (alert+column, vif_score)

    def save_model(self, modelfile, model):
        print 'Storing model to %s' % (modelfile,)
        pickle.dump(model, open(modelfile, 'wb'))

    def load_model(self, modelfile):
        print 'Loading model from %s' % (modelfile,)
        return pickle.load(open(modelfile))



class LinearModel(BaseModel):

    def baseline(self, X, y, features):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        #Baseline model
        model = LinearRegression()
        model.fit(X_train,y_train)
        print self.rmse(y_test, model.predict(X_test))

    def gridsearchCV(self, X, y, features):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        params = {'model__alpha': (.0001, .001, .005, .01, .05, .1, .5, 1.0, 5.0, 10.0) }
        pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                   ('model', Lasso())])

        grid_search = GridSearchCV(estimator=pipeline, param_grid=params, cv=10, scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=1, n_jobs=3)
        grid_search.fit(X_train,y_train)
        print grid_search.best_estimator_.__class__.__name__, -1*grid_search.best_score_, grid_search.best_estimator_.get_params()

        p = grid_search.best_estimator_.predict(X_test)
        print 'RMSE: %f' % self.rmse(y_test,p)

        model = grid_search.best_estimator_.named_steps['model']
        model_feature_importance(model.coef_, features) #, 'docs/linear_features1.png')

        transformer = SelectFromModel(model, prefit=True)
        best_coef = model.coef_[transformer.get_support(indices=True)]
        best_names = features[transformer.get_support(indices=True)]
        model_feature_importance(best_coef, best_names) #, 'docs/linear_features2.png')

        return grid_search.best_estimator_



class TreeModel(BaseModel):
    def gridsearchCV(self, X, y, features):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        params = {'model__n_estimators': (500, 1000, 1500),
                      'model__max_depth': (2, 4, 8),
                      'model__learning_rate': (.1, .01),
                      'model__loss': ('lad',) }
        pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                   ('model', ensemble.GradientBoostingRegressor())])

        grid_search = GridSearchCV(estimator=pipeline, param_grid=params, cv=5, scoring=make_scorer(mean_squared_error, greater_is_better=False), verbose=2, n_jobs=3)
        grid_search.fit(X_train,y_train)
        print grid_search.best_estimator_.__class__.__name__, -1*grid_search.best_score_, grid_search.best_estimator_.get_params()

        p = grid_search.best_estimator_.predict(X_test)
        print 'RMSE: %f' % self.rmse(y_test,p)

        # Plot feature importance
        regressor = grid_search.best_estimator_.named_steps['model']
        model_feature_importance(regressor.feature_importances_, features) #, 'docs/tree_features1.png')

        return grid_search.best_estimator_


