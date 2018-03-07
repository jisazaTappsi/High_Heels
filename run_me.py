#!/usr/bin/env python

"""Main"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import util
import text_analisis
import models
import copy
import result
import time


start = time.time()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# --------------------- ETL ---------------------

data_labeling = pd.DataFrame.from_csv('input-data-labeling.csv')
data_labels = pd.DataFrame.from_csv('input-data-labels.csv', parse_dates=False)

# only a fraction of data tested due to performance.
data_labeling = data_labeling.sample(frac=0.1, random_state=seed)

text_analisis.add_high_frequency_words(data_labeling, top_words=0.01)


num_features_dict = {'shop': 13, 'name': 15, 'description': 15}
util.hash_and_transform_x(data_labeling, num_features_dict)

# gets the level 1 out of the rest of labels.
data_labeling['label_1'] = data_labeling.apply(util.get_level1_label, args=('labels',), axis=1)

labels1, encoder1 = util.get_labels_and_encoder(data_labeling)


xtrain, xtest, ytrain, ytest = train_test_split(data_labeling, labels1, test_size=0.20, random_state=seed)


# before removing labels it copies them.
train_labels = copy.copy(xtrain['labels'])
test_labels = copy.copy(xtest['labels'])


util.add_frequency(data_labels, train_labels)


# remove labels from x.
xtrain.drop('labels', axis=1, inplace=True)
xtrain.drop('label_1', axis=1, inplace=True)
xtest.drop('labels', axis=1, inplace=True)
xtest.drop('label_1', axis=1, inplace=True)

# copies text data and removes it.
text_names_train = copy.copy(xtrain['name_text'])
text_names_test = copy.copy(xtest['name_text'])
text_description_train = copy.copy(xtrain['description_text'])
text_description_test = copy.copy(xtest['description_text'])

xtrain.drop('name_text', axis=1, inplace=True)
xtest.drop('name_text', axis=1, inplace=True)
xtrain.drop('description_text', axis=1, inplace=True)
xtest.drop('description_text', axis=1, inplace=True)


print('x_train.shape: ' + str(xtrain.shape))
print('x_test.shape: ' + str(xtest.shape))
print('len(y_train): ' + str(len(ytrain)))
print('len(y_test): ' + str(len(ytest)))


# --------------------- LEARNING ---------------------

# Level 1
test_level1, predicted_level1 = models.train_level1(xtrain, ytrain, xtest, ytest)


# Level 2
print('\nTraining level 2...')
true_positives_level2 = []  # this is a nested list by columns.
true_negatives_level2 = []  # this is a nested list by columns.
level = 2
names2 = util.get_labels_by_frequency(level, data_labels)
train_params = dict(model_function=models.level2_model,
                    epochs=12,
                    batch_size=5,
                    number_of_predictions=30,
                    parent_names=names2,
                    train_labels=train_labels,
                    test_labels=test_labels,
                    data_labels=data_labels)

data = (xtrain, xtest, text_names_train, text_names_test, text_description_train, text_description_test)
models.train_level_simple(data, train_params, true_positives_level2, true_negatives_level2, label_names=names2)
print('Finished training level 2\n')

# Level 3
print('\nTraining level 3...')
level = 3
util.get_labels_by_frequency(level, data_labels)
true_positives_level3 = []  # this is a nested list by columns.
true_negatives_level3 = []  # this is a nested list by columns.
names3 = util.get_labels_by_frequency(level, data_labels)
train_params = dict(model_function=models.level3_model,
                    epochs=12,
                    batch_size=5,
                    number_of_predictions=19,
                    parent_names='any',
                    train_labels=train_labels,
                    test_labels=test_labels,
                    data_labels=data_labels)

data = (xtrain, xtest, text_names_train, text_names_test, text_description_train, text_description_test)
models.train_level_simple(data, train_params, true_positives_level3, true_negatives_level3, label_names=names3)
print('Finished training level 3\n')


# Level 4
print('\nTraining level 4...')
level = 4
util.get_labels_by_frequency(level, data_labels)
true_positives_level4 = []  # this is a nested list by columns.
true_negatives_level4 = []  # this is a nested list by columns.
names4 = util.get_labels_by_frequency(level, data_labels)
train_params = dict(model_function=models.level4_model,
                    epochs=12,
                    batch_size=5,
                    number_of_predictions=30,
                    parent_names='any',
                    train_labels=train_labels,
                    test_labels=test_labels,
                    data_labels=data_labels)

data = (xtrain, xtest, text_names_train, text_names_test, text_description_train, text_description_test)
models.train_level_simple(data, train_params, true_positives_level4, true_negatives_level4, label_names=names4)
print('Finished training level 4\n')

# Summary results:
print('\nresults for levels 1,2,3 and 4:')
print('level 1')
print('test, accuracy(level 1): {}%'.format(str(result.match_percentage(test_level1, predicted_level1))))
print('level 2:')
result.print_total_covered(true_positives_level2)
result.print_uncovered(true_negatives_level2)
print('level 3:')
result.print_total_covered(true_positives_level3)
result.print_uncovered(true_negatives_level3)
print('level 4:')
result.print_total_covered(true_positives_level4)
result.print_uncovered(true_negatives_level4)

# -------------------------- BONUS --------------------------
result.print_multiple_covered(true_positives_level4)

print('total time spent: ' + str(time.time() - start))



