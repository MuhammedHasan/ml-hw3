import cPickle as pickle
from io import StringIO
import heapq

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier

number_words = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
number_mapper = {v: i for i, v in enumerate(number_words)}
reverse_number_map = dict(enumerate(number_words))

# For each section save your results in the variables specified
csv_data = unicode(open('dataset.csv', 'r').read().replace('?', ''))
col_name = open('feature_labels.txt', 'r').read().replace('\n', '').split(',')
df = pd.read_csv(StringIO(csv_data), names=col_name)

# section (a) save your list in the variable below
missing_features = list(len(df) - df.count())

# section (b) save your dataframe in the variable below
df_b = df.copy()

imp = Imputer(missing_values=np.nan, strategy="most_frequent", axis=1)
df_b['num-of-doors'] = df_b['num-of-doors'].map(number_mapper)
df_b['num-of-doors'] = imp.fit_transform([df_b['num-of-doors']])[0]
df_b['num-of-doors'] = df_b['num-of-doors'].map(reverse_number_map)

imp = Imputer(missing_values=np.nan, strategy="median", axis=0)
cols = ['horsepower', 'peak-rpm']
df_b[cols] = imp.fit_transform(df_b[cols].values)

imp = Imputer(missing_values=np.nan, strategy="mean", axis=0)
cols = ['normalized-losses', 'price', 'bore', 'stroke']
df_b[cols] = imp.fit_transform(df_b[cols].values)

# section (c) save your dataframe in the variable below
df_c = df_b.copy()
df_c['num-of-doors'] = df_c['num-of-doors'].map(number_mapper)
df_c['num-of-cylinders'] = df_c['num-of-cylinders'].map(number_mapper)

# section (d) save your dataframe in the variable below
df_d = df_c.copy()
df_d = pd.get_dummies(df_d)

# section (e) save your results in the variables below

collist = df_d.columns.tolist()[1:]


def accuracy_of_data(df, collist=collist):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        df_d[collist].values, df_d['symboling'].values,
        test_size=0.1, random_state=0)
    tree = DecisionTreeClassifier(
        criterion='entropy', max_depth=20, random_state=0)
    tree.fit(X_train, y_train)
    return tree.score(X_train, y_train), tree.score(X_test, y_test)

training_accuracy, test_accuracy = accuracy_of_data(df_d)

# print training_accuracy, test_accuracy
# section (f) save your list of tuples in the variable below
forest = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=20)
forest.fit(X_train, y_train)
feature_importance = zip(collist, forest.feature_importances_)

# section (g) save your result in the variable below
cheap = [
    [j[0] for j in heapq.nlargest(i, feature_importance, key=lambda s: s[1])]
    for i in range(1, len(feature_importance) + 1)
]
accuracies = [accuracy_of_data(df_d, i)[1] for i in cheap]

num_of_features = max(enumerate(accuracies), key=lambda x: x[1])[0] + 1

# print num_of_features


def visualise_line_graph():
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.ylim([0.45, 0.85])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    plt.show()

visualise_line_graph()

# section (h) use the function given below
neu_net = Pipeline([
    ('scaler', StandardScaler()),
    # ('reduce_dim', KernelPCA()),
    # ('clf', MLPClassifier(hidden_layer_sizes=(1000,),
    #                       solver='adam', random_state=0)),
    # ('clf', DecisionTreeClassifier(max_features=num_of_features, random_state=0))
    # ('clf', SVC(kernel='rbf', C=1000))
    # ('clf', KNeighborsClassifier())
    ('clf', RandomForestClassifier(n_estimators=1000, random_state=0))
])

print num_of_features

for _ in range(50):
    model = neu_net.fit(X_train, y_train)
    print accuracy_score(model.predict(X_train), y_train)
    print accuracy_score(model.predict(X_test), y_test)

#
# def custom_model_prediction(test_set):
#     """This should return the accuracy
#        Parameters
#        ----------
#        test_set: text file representing the test set with
#                  same format as dataset.txt without symboling column
#        Return
#        ------
#        predictions: type-list
#        """
#     predictions = []
#     return predictions
#
# # Ignore this section
# all_results = [missing_features, df_b, df_c, df_d,
#                (training_accuracy, test_accuracy),
#                feature_importance, num_of_features]
#
# pickle.dump(all_results, open('saved_results.p', 'wb'))

df.to_csv('df.csv')
df_b.to_csv('df_b.csv')
df_c.to_csv('df_c.csv')
df_d.to_csv('df_d.csv')
