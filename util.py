#!/usr/bin/env python

"""Utility functions"""

from sklearn.feature_extraction import FeatureHasher
import pandas as pd
import sklearn.preprocessing
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np


def get_level1_label(row, key):
    return row[key].split(',')[0]

assert get_level1_label({'label': 'a,b,c,d'}, 'label') == 'a'


def get_label(row, key):
    """Categorizes each row as having or not having the key"""
    return int(key in row.split(','))


def get_labels_with_array(labels, key):
    """Similar to get_label(), but for the labels array"""
    return [get_label(row, key) for row in labels]


def from_categorical(data):
    """Inverse function of np_utils.to_categorical() only works for 3 classes."""

    value_map = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}

    candidate = None
    solution = []

    for d in data:
        error = 1e5
        for k, v in value_map.items():
            #e = sum([abs(d[0] - v[0])])
            e = abs(d[0] - v[0]) + abs(d[1] - v[1]) + abs(d[2] - v[2])

            if e < error:
                error = e
                candidate = k

        solution.append(candidate)

    return solution

# nice input
assert from_categorical([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == [0, 1, 2]

# noisy input
assert from_categorical([[0.99, 0.01, 0.01], [0.1, 1.1, 0.1], [-.1, .4, 0.9]]) == [0, 1, 2]


def hash_column(data_labeling, column_name, num_features_dict, df):
    """uses the hash trick to encode many categorical values in few dimensions."""

    num_features = num_features_dict[column_name]

    hasher = FeatureHasher(n_features=num_features, non_negative=False, input_type='string')
    X_new = hasher.fit_transform(data_labeling[column_name])
    hashed_df = pd.DataFrame(X_new.toarray())

    for i in range(num_features):
        df[column_name + '_' + str(i)] = list(hashed_df[i])

    # Finally, remove old column
    data_labeling.drop(column_name, axis=1, inplace=True)


def get_input_dim(num_features_dict):
    return sum([v for k, v in num_features_dict.items()])


def hash_and_transform_x(data_labeling, num_features_dict):

    data_labeling.drop('brand', axis=1, inplace=True)
    hash_column(data_labeling, 'shop', num_features_dict=num_features_dict, df=data_labeling)
    hash_column(data_labeling, 'name', num_features_dict=num_features_dict, df=data_labeling)
    hash_column(data_labeling, 'description', num_features_dict=num_features_dict, df=data_labeling)

    # rescale price.
    data_labeling['price'] = sklearn.preprocessing.scale(data_labeling['price'])


def get_labels_and_encoder(data_labeling):
    """Returns the labels in the encoder. This is for multi-class."""
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(data_labeling['label_1'])
    encoded_lvl1 = encoder.transform(data_labeling['label_1'])
    # convert integers to dummy variables (i.e. one hot encoded)
    label1 = np_utils.to_categorical(encoded_lvl1)

    return label1, encoder


def flatten_list(l):
    """Make nested list, flat."""
    return [item for sublist in l for item in sublist]


def get_average(vector):
    """Just the average"""
    return sum(vector) / len(vector)


def scale_score_by_frequency(score, ytrain):
    """The score is scaled and quantized so that it has the same mean as y_train.
    :param score:
    :param ytrain"""
    specificity_multiplier = 1
    avg_score = get_average(ytrain)
    median_score = np.percentile(score, min(100, 100 * (1 - avg_score) * specificity_multiplier))
    return [int(e > median_score) for e in score]


def add_frequency(data_labels, train_labels):
    """Adds the frequency to each category of data_labels given train_data_labeling"""

    frequency = []
    for name in data_labels['name']:
        count = 0
        for categories in train_labels:
            count += int(name in categories.split(','))
        frequency.append(count)

    data_labels['frequency'] = frequency


def get_labels_by_frequency(level, data_labels):
    """Returns list of labels by descending frequency. filter and order by frequency"""

    # filter df
    filtered = data_labels[data_labels.level == level]#) & (df.D == 6)]
    return np.array(filtered.sort_values(['frequency'], ascending=[False])['name'])


def add_label_in_text(df, class_name, column_name, text):
    """Adds a column with 0 and 1 indicating whether it found any brother on the text and drops the text column"""
    lower_class = class_name.lower()
    df['label_in_' + column_name] = text.apply(lambda o: int(lower_class in o.lower()))


def get_brothers(label_name, data_labels):
    """Given the label name it gets the brothers"""
    row = data_labels[data_labels.name == label_name]['parent_tag_id'].values
    parent_tag_id = row[0]
    return data_labels[data_labels.parent_tag_id == parent_tag_id]['name'].values


def invert_nested_list(nested_list):
    return list(map(list, zip(*nested_list)))
