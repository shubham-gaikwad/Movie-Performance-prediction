
# coding: utf-8

# In[ ]:

import pandas as pd
import os


# In[ ]:

dir_path = '/Users/Aniket/Shubham'
movies_path = os.path.join(dir_path,'movies_found.csv')
video_stats = os.path.join(dir_path,'Videos_stats.csv')
video_info = os.path.join(dir_path,'videos_info.csv')
comment_sentiments = os.path.join(dir_path,'comment_sentiment.csv')
all_file = os.path.join(dir_path,'all.csv')


# In[ ]:

movies_cols = ['movieID', 'movie Name', 'budget', 'worldwise_gross_income']
movies_df = pd.read_csv(movies_path, delimiter = '|', error_bad_lines = False, index_col = False, names = movies_cols)
movies_df['income_ratio'] =  movies_df['worldwise_gross_income'].astype('float') /movies_df['budget'].astype('float')
movies_df = movies_df[~movies_df.income_ratio.isnull()]
movies_df.head(5)


# In[ ]:

## Normalize income_ration
from sklearn.preprocessing import MinMaxScaler

movies_df[['normalized_income_ratio']] = movies_df[['income_ratio']].apply(lambda x : MinMaxScaler().fit_transform(x))
movies_df.head(5)
#(movies_df.income_ratio -movies_df.income_ratio.mean()) / (movies_df.income_ratio.max() - movies_df.income_ratio.min())


# In[ ]:

videos_stats_cols = ['movieID', 'Trailer_ID', 'comment_count', 'view_count', 'favourite_count', 'dislike_count',  'like_count']
video_stats_df = pd.read_csv(video_stats, delimiter = '|', error_bad_lines = False, index_col = False, names = videos_stats_cols)
video_stats_df.head(5)


# In[ ]:

## Add up all counts
grouped_video_stats = video_stats_df.groupby('movieID')[['comment_count', 'view_count', 'favourite_count', 'dislike_count', 'like_count']].sum()
video_stats_df = grouped_video_stats.reset_index()
video_stats_df.head(5)


# In[ ]:

video_stats_df['movieID'] = video_stats_df.movieID.astype('int')
movies_df['movieID'] = movies_df.movieID.astype('int')
final_df1 = video_stats_df.set_index('movieID').join(movies_df.set_index('movieID'))
#video_stats_df.join(movies_df, rsuffix='_movies_df', lsuffix='_video_stats_df')
final_df1.head(10)


# In[ ]:

videos_info_cols = ['movieID', 'movie_Name', 'Day', 'Month', 'year_of_release', 'IMDB_movie_id']
videos_info_df = pd.read_csv(video_info, delimiter = '|', error_bad_lines = False, index_col = False, names = videos_info_cols)
videos_info_df.head(5)


# In[ ]:

videos_info_df['movieID'] = videos_info_df.movieID.astype('int')
final_df2 = final_df1.join(videos_info_df.set_index('movieID'))
#video_stats_df.join(movies_df, rsuffix='_movies_df', lsuffix='_video_stats_df')
final_df2.head(10)


# In[ ]:

comment_sentiments_cols = ['movieID', 'positive_comments', 'negative_comments', 'ratio_positive_total_comments']
comment_sentiments_df = pd.read_csv(comment_sentiments, delimiter = '|', error_bad_lines = False, index_col = False, names = comment_sentiments_cols)
comment_sentiments_df.head(5)


# In[ ]:

comment_sentiments_df['movieID'] = comment_sentiments_df.movieID.astype('int')
final_df3 = final_df2.join(comment_sentiments_df.set_index('movieID'))
#video_stats_df.join(movies_df, rsuffix='_movies_df', lsuffix='_video_stats_df')
final_df3.head(10)


# In[ ]:

all_file_cols = ['movieID', 'movie_name', 'actor1', 'follo1', 'actor2', 'follo2', 'actor3', 'follo3','actor4',                  'follo4', 'genre1', 'genre_freq_points1','genre2', 'genre_freq_points2', 'genre3', 'genre_freq_points3']
all_file_df = pd.read_csv(all_file, delimiter = '|', error_bad_lines = False, index_col = False, names = all_file_cols)
all_file_df.head(5)


# In[ ]:

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
all_file_df['movieID'] = all_file_df.movieID.astype('int')
final_df4 = final_df3.join(all_file_df.set_index('movieID'))
final_df4.head(10)


# In[ ]:

## Reset Index
final_feature = final_df4.copy()
final_feature = final_feature.reset_index()
final_feature.head(5)


# In[ ]:

len(final_feature[final_feature['normalized_income_ratio'] < 0.05])


# In[ ]:

def get_class_label(x):
    val = x[0]
    if val >= 0.2:
        return 1
    elif val >=0.05 and val < 0.2:
        return 0
    else:
        return -1


# In[ ]:

## Assign class labels
final_feature['class_label'] = final_feature[['normalized_income_ratio']].apply(lambda x : get_class_label(x), axis = 1 )
final_feature.head(5)


# In[ ]:

## Class label distribution
import matplotlib.pyplot as plt
colors = ['red','green', 'blue']
final_feature.class_label.value_counts().plot(kind = 'pie', colors = colors, autopct='%.2f')
plt.show()


# In[ ]:

## Remove data points without class_labels
print '\n Total movies extracted : {0}'.format(len(final_feature))
final_feature = final_feature[~final_feature.class_label.isnull()]
print '\n Number of movies that has class labels : {0}'.format(len(final_feature))


# In[ ]:

## Outlier in dataset
## Very few movies has normalized_income_ratio > 0.3
## So need to assign class label based on it
final_feature[final_feature['class_label'] == 1]
final_feature.normalized_income_ratio.hist()
plt.show()


# In[ ]:

## Data cleansing step : Convert to numbers
final_feature = final_feature.convert_objects(convert_numeric=True)
final_feature.head(5)


# In[ ]:

features_remove = ['movieID', 'movie Name', 'budget', 'worldwise_gross_income', 'income_ratio', 'normalized_income_ratio',                   'movie_Name', 'IMDB_movie_id', 'movie_name', 'positive_comments', 'negative_comments']

categorical_features1 = ['actor1', 'actor2', 'actor3', 'actor4', 'genre1', 'genre2','genre3', 'Day','Month','year_of_release']
features_remove.extend(categorical_features1)

categorical_features = []

numeric_features = ['comment_count', 'view_count', 'favourite_count', 'dislike_count', 'like_count', 'positive_comments',                   'negative_comments', 'ratio_positive_total_comments', 'follo1', 'follo2', 'follo3', 'follo4',                    'genre_freq_points1', 'genre_freq_points2', 'genre_freq_points3']


# In[ ]:

# final_feature.comment_count.unique()


# # PIPELINE UTILITY FUNCTIONS AND CLASSES
# 

# In[ ]:

from sklearn.base import TransformerMixin, BaseEstimator
class feature_removal(TransformerMixin, BaseEstimator):
    def __init__(self, columns = None):
        self.columns = columns
    
    def fit(self, X = None, y = None, **fit_args):
        return self
    
    def transform(self, X, **transform_args):
        copy_X = X.copy()
        #print copy_X.columns
        if isinstance(X, pd.DataFrame):
            if self.columns is not None:
                copy_X.drop(self.columns, axis = 1, inplace = True)
        else:
            print '\n Please pass Pandas DataFrame'
        return copy_X


# In[ ]:

def get_feature_and_label(dataframe):
    class_label = dataframe['class_label']
    features = dataframe.copy(deep = True)
    del features['class_label']
    return features, class_label


# In[ ]:

from sklearn.base import TransformerMixin, BaseEstimator
cols = 'xyz'
class one_hot_encoding(TransformerMixin, BaseEstimator):
    def __init__(self, columns = None):
        self.columns = columns
    
    def fit(self, X = None, y = None):
        return self
    
    def transform(self, X, y = None):
        global cols
        copy_X = X.copy()
        if isinstance(X, pd.DataFrame):
            if self.columns is not None:
                for col in self.columns:
                    prefix = col
                    prefix_sep = '_'
                    one_hot = pd.get_dummies(X[col], prefix = prefix, prefix_sep = prefix_sep)
                    if col + prefix_sep + prefix in one_hot.columns:
                        one_hot = one_hot.drop(col + prefix_sep + prefix, axis = 1) 
                    #print one_hot
                    copy_X = copy_X.join(one_hot)
                    copy_X = copy_X.drop(col, axis = 1)
            else:
                for col in list(X.columns.values):
                    prefix = col
                    prefix_sep = '_'
                    one_hot = pd.get_dummies(X[col], prefix = prefix, prefix_sep = prefix_sep)
                    if col + prefix_sep + prefix in one_hot.columns:
                        one_hot = one_hot.drop(col + prefix_sep + prefix, axis = 1) 
                    #print one_hot
                    copy_X = copy_X.join(one_hot)
                    copy_X = copy_X.drop(col, axis = 1)
        else:
            print '\n Please pass Pandas DataFrame'
        cols = copy_X.columns
        return copy_X


# In[ ]:

# ## Replace missing values
# from sklearn.preprocessing import Imputer

# removal_instance =  feature_removal(features_remove)
# df1 = removal_instance.fit_transform(final_feature)

# one_instance = one_hot_encoding(categorical_features)
# df2 = one_instance.fit_transform(df1)

# imp_instace = Imputer(axis=0, strategy='median')
# df3 = imp_instace.fit_transform(df2)

# for x in df3:
#     if len(x) != 26:
#         print x


# # BASLINE PIPELINE

# In[ ]:

from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import classification_report

# Seperate feature from class label
print '\n final_feature : {0}'.format(final_feature.shape)
X, Y = get_feature_and_label(final_feature)
print '\n X : {0}, Y : {1}'.format(X.shape, Y.shape)
# print '\n type(X) : {0}, type(Y) : {1}'.format(type(X), type(Y))

# Split data
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)
print '\n After splitting ....'
print '\n x_train : {0}, y_train : {1}'.format(x_train.shape, y_train.shape)


pipe_1 = Pipeline([
        ("feature_removal", feature_removal(features_remove)),
        ("one_hot_encoding",  one_hot_encoding(categorical_features)),
        ("imputer", Imputer(axis=0, strategy='median')),
        ("random_forest" , OneVsOneClassifier(RandomForestClassifier()))
    ])


kfold = KFold(n_splits = 5, shuffle = True)
model = pipe_1.fit(x_train, y_train)
# model_file_path = '/Users/Aniket/Appzen/myenv/Source/semanticzen/learned_models/random_forest_baseline.pkl'
# joblib.dump(model, model_file_path)
# print '\n model : {0}'.format(model)
# print '\n Model is dumped to : {0}'.format(model_file_path)


scores = cross_val_score(model,  # steps to convert raw messages into models
                         x_train,  # training data
                         y_train,  # training labels
                         cv=kfold,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )

print '\n Train result : cross_validation'
print '\n Mean : {0}, std : (+/-) {1}'.format(scores.mean(), scores.std())

trained_model = model.steps[3][1]
print '\n trained_model : {0}'.format(trained_model)

y_prediction = model.predict(x_test)
report = classification_report(y_test, y_prediction)
print '\n ---------- Classification Report ------------'
print report


# # GRID SEARCH

# In[ ]:

## Random Forest
# import os
# from sklearn import cross_validation
# from sklearn.model_selection import cross_val_score, KFold
# import sklearn.pipeline as pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# #from sklearn.svm import svc
# from sklearn.preprocessing import Imputer
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import confusion_matrix
# from sklearn.externals import joblib
# from sklearn.pipeline import Pipeline
# from sklearn import tree
# import pickle


# # Seperate feature from class label
# X, Y = get_feature_and_label(final_feature)
# print '\n X : {0}, Y : {1}'.format(X.shape, Y.shape)
# # print '\n type(X) : {0}, type(Y) : {1}'.format(type(X), type(Y))

# # Split data
# x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)
# # print '\n After splitting ....'
# print '\n x_train : {0}, y_train : {1}'.format(x_train.shape, y_train.shape)


# # Initialization

# pipe_1 = Pipeline([
#         ("feature_removal", feature_removal(features_remove)),
#         ("one_hot_encoding",  one_hot_encoding(categorical_features)),
#         ("imputer", Imputer(axis=0)),
#         ("random_forest" , RandomForestClassifier()),
#         ("SVM" , SVC())
#     ])


# # Grid Search
# parameters = dict(imputer__strategy=['median', 'mean'],
#                   random_forest__max_features=['sqrt', 'log2'],
#                   random_forest__n_estimators=[200],
#                   random_forest__min_samples_split=[2, 3, 4, 5, 10],
#                   random_forest__class_weight=[{-1:1, 0:4, 1:18},{-1:1,0:4, 1:36}],
#                   SVM__kernel=['linear', 'rbf']
#                  )

# kfold = KFold(n_splits = 5, shuffle = True)
# cv = GridSearchCV(
#                   pipe_1,
#                   param_grid = parameters, 
#                   refit=True,
#                   scoring='accuracy', 
#                   n_jobs = -1,
#                   cv=5
#                 )


# # Fit grid search with pipeline
# model = cv.fit(x_train, y_train)


# # Get feature imporatance
# print 'Best score: %0.3f' % model.best_score_
# print 'Best parameters set:'
# best_parameters = model.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print '\t%s: %r' % (param_name, best_parameters[param_name])

    
# # Get prediction
# y_prediction = cv.predict(x_test)

    
# # Get report
# report = classification_report( y_test, y_prediction )


# # Print report
# print report


# # Confusion matrix
# c_matrix = confusion_matrix(y_test, y_prediction, labels = [-1, 0, 1])
# print '\n Confusion matrix '
# print c_matrix
# print '(row=expected, col=predicted)'


# # Get Best Model
# best_estimator = model.best_estimator_
# print '\n best_estimator : '
# print best_estimator
# model = best_estimator.steps[3][1]
# print '\n Best estimator : {0}'.format(model)


