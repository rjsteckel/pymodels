# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import string
import pickle
import collections
from datetime import date

from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion



class DataFrameMissingRowRemover(TransformerMixin):
    """Removes any rows with a percentage of missing values
        greater than the pct given
    """
    def __init__(self, pct=.25):
        self.pct = pct

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        missing_counts = df.apply(lambda x: sum(x.isnull().values), axis = 1)
        missing_idx = np.where(missing_counts > (self.pct * df.shape[1]))[0]
        if(len(missing_idx) > 0):
            df = df.drop(df.index[[missing_idx]])
        return df


class DataFrameColumnRemover(TransformerMixin):
    """ Simply removes provided columns
    """
    def __init__(self, column_names=None):
        self.column_names = column_names

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        for column in self.column_names:
            df = df.drop(column, 1)
        return df


class DataFrameImputer(TransformerMixin):
    """Basic imputation
        Categorical variable are imputed with most frequent category
        Numeric variables are imputed with median value
    """
    def __init__(self, column_names=None):
        self.column_names = column_names

    def fit(self, X, y=None):
        columns_to_impute = self.column_names if self.column_names else [ c for c in X ]
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].median() for c in columns_to_impute], index=X[columns_to_impute].columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


class DataFrameCategoricalEncoder(TransformerMixin):
    """ Encodes categorical columns to binary columns (one for each category)
       (i.e  Dataframe[Color]      -->     Color-red, Color-green
              red                               1           0
              red                               1           0
              green                             0           1)
    """
    def __init__(self, categorical_names):
        if not isinstance(categorical_names, collections.Iterable):
            raise Exception('categorical names should be a list of strings')
        self.categorical_names = categorical_names

    def fit(self, df, y=None):
        self.vec = DictVectorizer()
        self.vec.fit(df[self.categorical_names].to_dict(orient='records'))
        return self

    def transform(self, df, y=None):
        vec_data = pd.DataFrame(self.vec.transform(df[self.categorical_names].to_dict(orient='records')).toarray())
        vec_data.columns = self.vec.get_feature_names()
        vec_data.index = df.index
        df = df.drop(self.categorical_names, axis=1)
        df = df.join(vec_data)
        return df


class DataFrameMYDateEncoder(TransformerMixin):
    """Encodes dataframe date columns that are strings of the form <abbreviated month>-<2 digit year>
        It handles the month and 2-digit year in either order
        (i.e. Jan-99 or 99-Jan)
    """
    def __init__(self, date_names):
        if not isinstance(date_names, collections.Iterable):
            raise Exception('date names should be a list of strings')
        self.date_names = date_names

    def correctMonthDashYearDateStr(d):
        tokens = d.split('-')
        if len(tokens) != 2:
            raise Exception("bad str date %s" % str(d))
        year = tokens[0] if tokens[0].isdigit() else tokens[1]
        year = year if len(year) == 2 else '0'+year
        year = '20'+year if int(year) < int(str(date.today().year)[2:]) else '19'+year
        month = tokens[0] if not tokens[0].isdigit() else tokens[1]
        return '%s-%s' % (month, year)

    def fit(self, df, y=None):
        return self

    def correctMonthDashYearDateStr(self, d):
        tokens = d.split('-')
        if len(tokens) != 2:
            raise Exception("bad str date %s" % str(d))
        year = tokens[0] if tokens[0].isdigit() else tokens[1]
        year = year if len(year) == 2 else '0'+year
        year = '20'+year if int(year) < int(str(date.today().year)[2:]) else '19'+year
        month = tokens[0] if not tokens[0].isdigit() else tokens[1]
        return '%s-%s' % (month, year)

    def transform(self, df, y=None):
        for dn in self.date_names:
            if df[dn].isnull().sum() > 0:
                raise Exception("NaN values detected. Impute first?")
            df[dn] = df[dn].astype('str').map(lambda x: self.correctMonthDashYearDateStr(x))
            df[dn] = pd.to_datetime(df[dn], format='%b-%Y')
        return df



class DataFrameStrFormatRemover(TransformerMixin):
    """Converts numeric str columns that have have some sort of formatting
        (i.e. '$12.34'  ->  12.34)
    """
    def __init__(self, str_names, chars_to_remove=[',', '%', '$', ' ', string.ascii_lowercase]):
        if not isinstance(str_names, collections.Iterable):
            raise Exception('str names should be a list of strings')
        self.str_names = str_names
        self.chars_to_remove = chars_to_remove

    def strToNumeric(self, df, column, remove_chars, new_dtype):
        for remove_char in remove_chars:
            df[column] = df[column].map(lambda s: np.nan if (s is None or s == np.nan or pd.isnull(s)) else str(s).translate(None, ''.join(self.chars_to_remove)))
        df[column] = df[column].apply(pd.to_numeric, args=('coerce',))
        df[column] = df[column].astype(new_dtype)
        return df

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        for name in self.str_names:
            print 'Cleaning %s' % name
            df = self.strToNumeric(df, name, self.chars_to_remove, 'float')
        return df


