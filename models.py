from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import util
import result
import numpy as np


# define level_1 model
def level_1_model(num_dims):
    model = Sequential()
    model.add(Dense(4, input_dim=num_dims, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define level_2 model
def level2_model(num_dims):
    model = Sequential()
    model.add(Dense(4, input_dim=num_dims, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


# define level_3 model
def level3_model(num_dims):
    model = Sequential()
    model.add(Dense(4, input_dim=num_dims, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


# define level_4 model
def level4_model(num_dims):
    model = Sequential()
    model.add(Dense(4, input_dim=num_dims, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def cross_validate(model, epochs, batch_size, num_dimensions, seed, xtrain_array, ytrain_array):
    """simple cross validation using keras and sklearn wrapper"""

    estimator = KerasClassifier(build_fn=model,
                                nb_epoch=epochs,
                                batch_size=batch_size,
                                num_dims=num_dimensions)

    kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

    # predicts label level 1.
    results = cross_val_score(estimator, xtrain_array, ytrain_array, cv=kfold)

    print("Train Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


def train_model(params, data):
    """Trains model and returns it"""
    xtrain_array, ytrain_array, xtest_array, ytest_array = data

    model_function = params['model_function']
    num_dimensions = params['num_dimensions']
    epochs = params['epochs']
    batch_size = params['batch_size']

    m = model_function.__call__(*[num_dimensions])
    m.fit(xtrain_array, ytrain_array, nb_epoch=epochs, batch_size=batch_size, verbose=0)

    scores = m.evaluate(xtest_array, ytest_array)
    print(" train accuracy: %s: %.2f%%" % (m.metrics_names[1], scores[1] * 100))
    return m


def train_model_and_eval(lvl2_label_name, data, true_positives_all, true_negatives_all, params):
    """Will train, evaluate and update the true positive nested_list"""

    (xtrain_array, ytrain_array, xtest_array, y_test) = data

    """
    models.cross_validate(model=models.level2_model,
                          epochs=epochs,
                          batch_size=batch_size,
                          num_dimensions=num_dimensions,
                          seed=seed,
                          xtrain_array=xtrain_array,
                          ytrain_array=y_train)
    """
    # --------------------- EVAL RESULTS ---------------------

    m2 = train_model(params=params,
                     data=(xtrain_array, ytrain_array, xtest_array, y_test))

    flatten_test_prediction = util.flatten_list(m2.predict(xtest_array))

    scaled_prediction = util.scale_score_by_frequency(flatten_test_prediction, ytrain_array)

    result.print_partial_results(lvl2_label_name, y_test, scaled_prediction, true_positives_all, true_negatives_all)


def train_level_simple(data, train_params, true_positives_all, true_negatives_all, label_names):
    """trains models for an entire level"""

    # data is unpacked, altered and then repacked.
    (x_train, x_test, text_names_train, text_names_test, text_description_train, text_description_test) = data

    for prediction_num in range(min(train_params['number_of_predictions'], len(label_names))):

        print('training prediction_num: {}'.format(prediction_num))

        label_name = label_names[prediction_num]

        # add connection between label name and text.
        util.add_label_in_text(x_train, label_name, 'name_text', text_names_train)
        util.add_label_in_text(x_test, label_name, 'name_text', text_names_test)
        util.add_label_in_text(x_train, label_name, 'description_text', text_description_train)
        util.add_label_in_text(x_test, label_name, 'description_text', text_description_test)

        train_params['num_dimensions'] = x_train.shape[1]

        xtrain_array = np.array(x_train.values.tolist())
        xtest_array = np.array(x_test.values.tolist())

        y_train = train_params['train_labels'].apply(util.get_label, args=(label_name,))
        y_test = train_params['test_labels'].apply(util.get_label, args=(label_name,))
        ytrain_array = np.array(y_train.values.tolist())
        ytest_array = np.array(y_test.values.tolist())

        data = (xtrain_array, ytrain_array, xtest_array, ytest_array)
        train_model_and_eval(label_name, data, true_positives_all, true_negatives_all, train_params)


def train_level1(xtrain, ytrain, xtest, ytest):
    """Trains a multiclass classifier for level 1"""
    epochs = 25
    batch_size = 5

    num_dimensions = xtrain.shape[1]
    xtrain_array = np.array(xtrain.values.tolist())
    xtest_array = np.array(xtest.values.tolist())

    """
    models.cross_validate(model=models.level_1_model,
                          epochs=epochs,
                          batch_size=batch_size,
                          num_dimensions=num_dimensions,
                          seed=seed,
                          xtrain_array=xtrain_array,
                          ytrain_array=y_train)
    """

    # --------------------- RESULTS ---------------------

    m1 = level_1_model(num_dimensions)
    m1.fit(xtrain_array, ytrain, nb_epoch=epochs, batch_size=batch_size)

    # evaluate the model
    predicted = m1.predict(xtest_array)

    predicted_level1 = util.from_categorical(predicted)
    test_level1 = util.from_categorical(ytest)

    return test_level1, predicted_level1
