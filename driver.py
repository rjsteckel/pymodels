# -*- coding: utf-8 -*-
import pandas as pd
from regression.model import *
from transform.df_transform import *


def derive_custom_features(df):
    #loan experience
    df['loan_experience'] = (df['loan_date'] - df['first_creditline']) / np.timedelta64(1, 'D')
    df = df.drop('loan_date', 1)
    df = df.drop('first_creditline', 1)

    #funded proportion of amount loaned
    df['funded_proportion'] = df['investor_funded_portion']/df['amount_funded']
    df = df.drop(['investor_funded_portion', 'amount_funded'], 1)
    return df


def dataframeToXY(df):
    X = df.drop('interest_rate', 1)
    y = df['interest_rate'].values
    feature_names = df.columns[1:]
    assert(X.shape[1] == len(feature_names))
    print 'Loaded %dx%d' % (X.shape[0], X.shape[1])
    return (X, y, feature_names)



if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1500)

    df = pd.read_csv('data/train.csv')
    df.head()

    date_cols = ['loan_date', 'first_creditline']
    categorical_names = ['loan_grade','number_of_payments','years_employed','home_ownership','income_verification','loan_category','borrower_state','initial_status']
    remove_cols = ['loan_id', 'borrower', 'employer', 'loan_title', 'loan_reason', 'loan_subgrade', 'zip_code3']
    strColumns = ['interest_rate', 'amount_requested', 'amount_funded', 'investor_funded_portion', 'line_utilization_rate', 'number_of_payments']

    clean_pipeline = Pipeline([('missing', DataFrameMissingRowRemover(.25)),
                               ('remover', DataFrameColumnRemover(remove_cols)),
                               ('fmt_remove', DataFrameStrFormatRemover(strColumns)),
                               ('imputer', DataFrameImputer()),
                               ('encoder', DataFrameCategoricalEncoder(categorical_names)),
                               ('date_enc', DataFrameMYDateEncoder(date_cols)),
                               ('experience', FunctionTransformer(derive_custom_features, validate=False))])

    print 'Fitting clean pipeline'
    clean_pipeline.fit(df)

    print 'Cleaning training data'
    df = clean_pipeline.transform(df)
    X, y, feature_names = dataframeToXY(df)

    #----Part 1------
    #Train and save linear model
    print 'Running baseline model'
    linear = LinearModel()
    linear.baseline(X, y, feature_names)
    print 'Running regularized model'
    linear_model = linear.gridsearchCV(X, y, feature_names)
    linear.save_model('models/linear.pkl', linear_model)

    #Train and save tree model
    print 'Running tree model'
    tree = TreeModel()
    tree_model = tree.gridsearchCV(X, y, feature_names)
    tree.save_model('models/tree.pkl', tree_model)

    #----Part 2------
    #Load and transform the holdout data
    print 'Cleaning test data'
    test_df = pd.read_csv('data/test.csv')
    test_df = clean_pipeline.transform(test_df)
    test_df['interest_rate'] = np.nan
    testX, testy, feature_names = dataframeToXY(df)

    #Load and run models on holdout dat
    print 'Predicting on test with linear model'
    stored_linear_model = linear.load_model('models/linear.pkl')
    linear_predicted = stored_linear_model.predict(testX)

    print 'Predicting on test with tree model'
    stored_tree_model = tree.load_model('models/tree.pkl')
    tree_predicted = stored_tree_model.predict(testX)

    print 'Saving results'
    pd.DataFrame({'LinearPredictions': linear_predicted, 'TreePredictions': tree_predicted}).to_csv('data/Results_From_Ryan_Steckel.csv', index=False)