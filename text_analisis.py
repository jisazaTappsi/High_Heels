import operator
import copy


def word_frequency(text_list, min_len=3):
    """returns list with ordered tuples ordered by descending word frequency count. Filters for len(word) < min_len"""

    frequency_dict = {}

    for e in text_list:
        words = set(str(e).split())

        for w in words:

            if len(w) > min_len:
                if w in frequency_dict.keys():
                     frequency_dict[w] += 1
                else:
                     frequency_dict[w] = 1

    sorted_x = sorted(frequency_dict.items(), key=operator.itemgetter(1))
    return list(reversed(sorted_x))


def pick_words(row, key, name_words):
    """Selects words that match with list."""
    return [e for e in name_words if e in row[key]]


def add_high_frequency_words(data_labeling, top_words=0.01):
    """Transforms the columns 'name' and 'description' to be arrays of only high frequency words. and renames original
    text fields"""
    # frequency count for words longer than 3 letters.
    name_words = word_frequency(data_labeling['name'], min_len=3)
    description_words = word_frequency(data_labeling['description'], min_len=3)

    # filter for most common words:
    name_words = name_words[:int(len(name_words)*top_words)]

    # remove frequency:
    name_words = [e for e, fn in name_words]
    description_words = description_words[:int(len(description_words)*top_words)]
    # remove frequency:
    description_words = [e for e, fd in description_words]

    data_labeling['name_text'] = copy.deepcopy(data_labeling['name'])
    data_labeling['description_text'] = copy.deepcopy(data_labeling['description'])

    data_labeling['name'] = data_labeling.apply(pick_words, args=('name', name_words), axis=1)
    data_labeling['description'] = data_labeling.apply(pick_words, args=('description', description_words), axis=1)
