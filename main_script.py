import cPickle as pickle
from io import StringIO
import heapq
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

number_words = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
number_mapper = {v: i for i, v in enumerate(number_words)}
reverse_number_map = dict(enumerate(number_words))
imp_mode, imp_median, imp_mean, vec = False, False, False, False

csv_data = unicode(open('dataset.txt', 'r').read().replace('?', ''))
col_name = open('feature_labels.txt', 'r').read(
).replace('\n', '').split(',')
df = pd.read_csv(StringIO(csv_data), names=col_name)


# section (a) save your list in the variable below
missing_features = list(len(df) - df.count())


# section (b) save your dataframe in the variable below
df_b = df.copy()

imp_mode = Imputer(missing_values=np.nan, strategy="most_frequent", axis=1)
df_b['num-of-doors'] = df_b['num-of-doors'].map(number_mapper)
df_b['num-of-doors'] = imp_mode.fit_transform([df_b['num-of-doors']])[0]
df_b['num-of-doors'] = df_b['num-of-doors'].map(reverse_number_map)

cols = ['horsepower', 'peak-rpm']
imp_median = Imputer(missing_values=np.nan, strategy="median", axis=0)
df_b[cols] = imp_median.fit_transform(df_b[cols].values)

cols = ['normalized-losses', 'price', 'bore', 'stroke']
imp_mean = Imputer(missing_values=np.nan, strategy="mean", axis=0)
df_b[cols] = imp_mean.fit_transform(df_b[cols].values)


# section (c) save your dataframe in the variable below
df_c = df_b.copy()
df_c['num-of-doors'] = df_c['num-of-doors'].map(number_mapper)
df_c['num-of-cylinders'] = df_c['num-of-cylinders'].map(number_mapper)


# section (d) save your dataframe in the variable below
df_d = df_c.copy()

cols = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels',
        'engine-location', 'engine-type', 'fuel-system']
vec = DictVectorizer()

vec_data = pd.DataFrame(
    vec.fit_transform(df_d[cols].to_dict('records')).toarray())
vec_data.columns = [i.replace('=', '_') for i in vec.get_feature_names()]
vec_data.index = df_d.index

df_d = df_d.drop(cols, axis=1)
df_d = df_d.join(vec_data)


# section (e) save your results in the variables below
collist = df_d.columns.tolist()
collist.remove('symboling')


def accuracy_of_data(df, collist=collist):
    X_train, X_test, y_train, y_test = train_test_split(
        df_d[collist].values, df_d['symboling'].values,
        test_size=0.3, random_state=0)
    tree = DecisionTreeClassifier(
        criterion='entropy', max_depth=20, random_state=0)
    tree.fit(X_train, y_train)
    return tree.score(X_train, y_train), tree.score(X_test, y_test)

training_accuracy, test_accuracy = accuracy_of_data(df_d)

# section (f) save your list of tuples in the variable below
forest = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=20)
X_train, X_test, y_train, y_test = train_test_split(
    df_d[collist].values, df_d['symboling'].values,
    test_size=0.3, random_state=0)
forest.fit(X_train, y_train)
feature_importance = zip(collist, forest.feature_importances_)

# section (g) save your result in the variable below
cheap = [
    [j[0] for j in heapq.nlargest(i, feature_importance, key=lambda s: s[1])]
    for i in range(1, len(feature_importance) + 1)
]
accuracies = [accuracy_of_data(df_d, i)[1] for i in cheap]

num_of_features = max(enumerate(accuracies), key=lambda x: x[1])[0] + 1


def visualise_line_graph():
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.ylim([0.45, 0.85])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    plt.show()


# section (h) use the function given below


def train_my_model():
    if os.path.isfile('model.p'):
        return pickle.load(open('model.p', 'rb'))

    neu_net = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=1000, random_state=0))
    ])
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    modelp = neu_net.fit(X, y)

    pickle.dump(modelp, open('model.p', 'wb'))
    return modelp


modelx = train_my_model()


def custom_model_prediction(test_set):
    """This should return the accuracy
       Parameters
       ----------
       test_set: text file representing the test set with
                 same format as dataset.txt without symboling column
       Return
       ------
       predictions: type-list
       """

    csv_data = unicode(open(test_set, 'r').read().replace('?', ''))
    col_name = open('feature_labels.txt', 'r').read() \
        .replace('\n', '').split(',')[1:]
    df_x = pd.read_csv(StringIO(csv_data), names=col_name)

    # section (b) save your dataframe in the variable below
    df_b_x = df_x.copy()
    df_b_x['num-of-doors'] = df_b_x['num-of-doors'].map(number_mapper)
    df_b_x['num-of-doors'] = imp_mode.transform([df_b_x['num-of-doors']])[0]
    df_b_x['num-of-doors'] = df_b_x['num-of-doors'].map(reverse_number_map)

    cols = ['horsepower', 'peak-rpm']
    df_b_x[cols] = imp_median.transform(df_b_x[cols].values)

    cols = ['normalized-losses', 'price', 'bore', 'stroke']
    df_b_x[cols] = imp_mean.transform(df_b_x[cols].values)

    # section (c) save your dataframe in the variable below
    df_c_x = df_b_x.copy()
    df_c_x['num-of-doors'] = df_c_x['num-of-doors'].map(number_mapper)
    df_c_x['num-of-cylinders'] = df_c_x['num-of-cylinders'].map(number_mapper)

    # section (d) save your dataframe in the variable below
    df_d_x = df_c_x.copy()

    cols = ['make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels',
            'engine-location', 'engine-type', 'fuel-system']

    vec_data = pd.DataFrame(
        vec.transform(df_d_x[cols].to_dict('records')).toarray())
    vec_data.columns = [i.replace('=', '_') for i in vec.get_feature_names()]
    vec_data.index = df_d_x.index

    df_d_x = df_d_x.drop(cols, axis=1)
    df_d_x = df_d_x.join(vec_data)

    return modelx.predict(df_d_x.values)

# Ignore this section
all_results = [missing_features, df_b, df_c, df_d,
               (training_accuracy, test_accuracy),
               feature_importance, num_of_features]

pickle.dump(all_results, open('saved_results.p', 'wb'))
