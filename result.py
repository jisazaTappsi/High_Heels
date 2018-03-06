import util


def get_average(vector):
    """Just the average"""
    return sum(vector) / len(vector)


def match_percentage(y1, y2):
    """Gets the total percentage between array y1 and y2"""
    return sum([int(e1 == e2) for e1, e2 in zip(y1, y2)])/len(y1)*100


assert match_percentage([0, 1, 2, 2], [0, 2, 1, 2]) == 50.0


def true_positive(test, pred):
    """True positives."""
    return int(test == 1 and pred == 1)


def true_negative(test, pred):
    """True negatives."""
    return int(test == 0 and pred == 0)


def get_true_positives_vector(true_y, prediction):
    """Gets the overlap for which true positives hold."""
    return [true_positive(test, pred) for test, pred in zip(true_y, prediction)]


def get_true_negative_vector(true_y, prediction):
    """Gets the overlap for which true positives hold."""
    return [true_negative(test, pred) for test, pred in zip(true_y, prediction)]


def get_cover_percent(true_y, prediction):
    """Of the total labels what percentage had a correct prediction?"""
    return get_average(get_true_positives_vector(true_y, prediction)) * 100


def get_max_cover(true_y):
    """Gets the maximum cover percentage or if all positive true values are considered."""
    return get_average(true_y) * 100


def reduce_cover(nested_list):
    """If at least 1 item on the row of the inner list is 1 then the reduced list is 1. The input is a nested_list
    by columns."""

    # inverts the nested list from a list containing columns to a one containing rows.
    inverted_list = util.invert_nested_list(nested_list)
    return [int(any(l)) for l in inverted_list]


def reduce_multiple_cover(nested_list):
    """If at least 2 items on the row of the inner list are 1 then the reduced list is 1. The input is a nested_list
    by columns."""

    # inverts the nested list from a list containing columns to a one containing rows.
    inverted_list = util.invert_nested_list(nested_list)
    return [int(sum(l) >= 2) for l in inverted_list]

assert reduce_cover([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) == [1, 1, 0]


def reduce_uncover(nested_list):
    """If all items on the row of the inner list are 1 then the reduced list is 1. The input is a nested_list
    by columns."""

    # inverts the nested list from a list containing columns to a one containing rows.
    inverted_list = list(map(list, zip(*nested_list)))
    return [int(all(l)) for l in inverted_list]


assert reduce_uncover([[1, 1, 0], [1, 0, 0], [1, 1, 0]]) == [1, 0, 0]


def to_percent(vector):
    """From a 0 anf 1 vector to the percent"""
    return get_average(vector) * 100


def get_total_cover_percent(true_positives_nested):
    """Given nested list of true positives gets the cover percent, of at least 1 of them."""
    return to_percent(reduce_cover(true_positives_nested))


def get_multiple_cover_percent(true_positives_nested):
    """Given nested list of true positives gets the cover percent, of at least 2 of them."""
    return to_percent(reduce_multiple_cover(true_positives_nested))


def get_uncover_percent(true_negatives_nested):
    """Given nested list of true negatives gets the cover percent, of all of them."""
    return to_percent(reduce_uncover(true_negatives_nested))


def print_total_covered(true_positives):
    print('total covered so far: {}%'.format(get_total_cover_percent(true_positives)))


def print_multiple_covered(true_positives):
    print('multiple cover: {}%'.format(get_multiple_cover_percent(true_positives)))


def print_uncovered(true_negatives):
    print('total uncovered correctly: {}%'.format(get_uncover_percent(true_negatives)))


def print_partial_results(label_name, true_y, prediction, true_positives_all, true_negatives_all):
    """just a print"""

    print('test, accuracy: {}%'.format(match_percentage(true_y, prediction)))
    percentage_cover = get_cover_percent(true_y, prediction)
    print('test on {0}, cover percentage: {1}%, from {2}% possible'.format(label_name,
                                                                           percentage_cover,
                                                                           get_max_cover(true_y)))

    # true positives
    true_positives_all.append(get_true_positives_vector(true_y, prediction))
    print_total_covered(true_positives_all)

    # true negatives
    true_negatives_all.append(get_true_negative_vector(true_y, prediction))
    print_uncovered(true_negatives_all)
