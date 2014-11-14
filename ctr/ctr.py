__author__ = 'acpigeon'

import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import classification_report
from datetime import datetime


def get_data(train=True):
    """Read data in from the file and yield row by row"""

    if train:
        #path = '../data/train_rev1_short_short.csv'  # 10 rows
        path = '../data/train_rev1_short.csv'  # 1000 rows
        #path = '/Users/acpigeon/Documents/kaggle_avazu_ctr_data/originalData/train_rev1.csv'  # ~140M rows
    else:
        path = '../data/test_rev1_short.csv'

    it = enumerate(open(path))
    it.next()  # burn the header row
    x = [0] * 25  # initialize row container
    for ix, line in it:
        #print ix, line
        for ixx, f in enumerate(line.strip().split(',')):

            # Record sample id
            if ixx == 0:
                sample_id = f

            # If this is the training data, record the dependent variable
            elif ixx == 1 and train:
                click = f

            # Use the hashing trick to one hot encode categorical features
            # One hot encode everything because we don't know what most of these features are yet
            # None of the features appear continuous, so this is ok for now
            else:
                if ixx not in [0, 1]:  # only return data elements, not ids or dependent variables
                    #  bonus: change 20 to like 4 or 5 and compare hashed row to one-hotted row, hash collisions!
                    x[ixx - 2] = abs(hash(str(ixx) + '_' + f)) % (2 ** 20)  # offset to account for removing id and class

        yield [sample_id, x, click] if train else (sample_id, x)


def get_len_feature_array():
    """Iterate through all the examples and set the number of necessary columns"""
    columns = 0
    for ids, rows, classes in get_data():
        row_max = max(rows)
        if row_max > columns:
            columns = row_max
    return columns


def train_test_split(cols):
    """Mark 20% of the samples for cross validation."""
    for id, row, cls in get_data():
        # print row
        this_row = np.zeros(cols + 1)
        for i in row:
            this_row[i] = 1
        sorting_hat = random.choice([1, 2, 3, 4, 5])  # pick 1-5 randomly
        if sorting_hat == 5:  # every fifth element goes to xval
            yield (0, id, this_row, cls)
        else:
            yield (1, id, this_row, cls)


if __name__ == '__main__':
    start = datetime.now()

    # record this value so we don't have to get it every run on the large trains...
    columns = get_len_feature_array()
    #cols = 1048575  # full train set, we can cheat on the full run and save about 18 minutes

    clf = linear_model.SGDClassifier(loss='log')

    # Set up params for batching
    train_batch_size = 100
    current_batch = 1
    cross_val_sample_ids, cross_val_samples, cross_val_sample_classes = [], [], []
    train_sample_ids, train_samples, train_sample_classes = [], [], []

    # Iterate through samples
    for train, sample_id, sample, sample_class in train_test_split(columns):
        if train:
            #print "train: {0}, {1}, {2}".format(str(sample_id), str(sample), str(sample_class))
            if len(train_sample_ids) < train_batch_size:
                train_sample_ids.append(sample_id)
                train_samples.append(sample)
                train_sample_classes.append(sample_class)
            else:
                clf.partial_fit(train_samples, train_sample_classes, classes=np.unique(train_sample_classes))

                # Reset batch containers
                cross_val_sample_ids, cross_val_samples, cross_val_sample_classes = [], [], []
                train_sample_ids, train_samples, train_sample_classes = [], [], []

                # Update progress
                print "Batch {} finished...".format(current_batch)
                current_batch += 1
        else:
            cross_val_sample_ids.append(sample_id)
            cross_val_samples.append(sample)
            cross_val_sample_classes.append(sample_class)
            #print "xval: {0}, {1}, {2}".format(str(sample_id), str(sample), str(sample_class))

    # Cross validation
    predictions = clf.predict(cross_val_samples)

    print classification_report(cross_val_sample_classes, predictions)

    print "Done, elapsed time: {}".format(str(datetime.now() - start))