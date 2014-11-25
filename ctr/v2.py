__author__ = 'acpigeon'

import numpy as np
import random
import csv
from itertools import islice
from sklearn import linear_model
from sklearn.metrics import classification_report
from datetime import datetime


def get_len_feature_array():
    """Iterate through all the examples and set the number of necessary columns"""
    cols = 0
    for row, cls in row_hasher():
        row_max = max(row)
        if row_max > cols:
            cols = row_max
    return cols


def row_hasher():
    """Takes an iterable of sample and one-hot-encodes them."""

    #path = '../data/train_10.csv'  # 10 rows
    #path = '../data/train_100.csv'  # 100 rows
    #path = '../data/train_1k.csv'  # 1000 rows
    #path = '../data/train_10k.csv'  # 10000 rows
    path = '../data/train_100k.csv'  # 100000 rows
    #path = '/Users/acpigeon/Documents/kaggle_avazu_ctr_data/originalData/train.csv'  # ~40M rows

    it = open(path)
    it.next()

    x = [0] * 25  # initialize row container once
    for l in it:
        this_start_time = datetime.now()
        for ixx, f in enumerate(l.strip().split(',')):

            # Record sample id, only need this for test set
            #if ixx == 0 and not train:
            #    sample_id = f

            # If this is the training data, record the dependent variable
            if ixx == 1:
                cls = int(f)

            # Use the hashing trick to one hot encode categorical features
            # One hot encode everything because we don't know what most of these features are yet
            # None of the features appear continuous, so this is ok for now
            else:
                if ixx not in [0, 1]:  # only return data elements, not ids or dependent variables
                    #  bonus: change 20 to like 4 or 5 and compare hashed row to one-hotted row, hash collisions!
                    x[ixx - 2] = abs(hash(str(ixx) + '_' + f)) % (2 ** 20)  # offset to account for removing id and class
        yield (x, cls)
        #print "row_hasher timing: {}".format(str(datetime.now() - this_start_time))


def to_dense(iterable, columns):
    """ Takes an iterable of one-hot encoded values and the number of variables and returns a dense row."""
    for i in iterable:
        this_start_time = datetime.now()
        this_row = np.zeros(columns + 2)  # +2 for the class and transition from sparse to dense
        #print sorted(i[0]), max(i[0])
        #print i[1]
        for j in i[0]:
            this_row[j] = 1
            this_row[-1] = i[1]
        yield this_row
        #print "to_dense timing: {}".format(str(datetime.now() - this_start_time))


def train_test_split(iterable):
    """Write 20% of the samples to a file for cross validation."""
    #writer = csv.writer(open('xval.csv', 'w'))
    for row in iterable:
        this_start_time = datetime.now()
        sorting_hat = random.choice([1, 2, 3, 4, 5])  # pick 1-5 randomly
        if sorting_hat == 5:  # every fifth element goes to xval
            #writer.writerow(row)
            continue  # only use this to complete the circuit if we're not writing anything out. we still lose xval rows
        else:
            yield row
            #print "train_test_split timing: {}".format(str(datetime.now() - this_start_time))


def grouper(n, iterable):
    """Chunk output from generator."""
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        this_start_time = datetime.now()
        yield np.array(piece)
        piece = list(islice(i, n))
        #print "grouper timing: {}\n".format(str(datetime.now() - this_start_time))


if __name__ == '__main__':
    start = datetime.now()
    batch_size = 1000

    # record this value so we don't have to get it every run on the large trains...
    #columns = get_len_feature_array()
    columns = 1048575  # full train set, we can cheat on the full run and save about 18 minutes

    clf = linear_model.SGDClassifier(loss='log', n_jobs=5, verbose=0)

    hashed_lines = row_hasher()
    dense_lines = to_dense(hashed_lines, columns)
    train_lines = train_test_split(dense_lines)
    grouped_set = grouper(batch_size, train_lines)

    current_batch = 1
    for batch in grouped_set:
        batch_start = datetime.now()

        if current_batch == 1:  # first run, use fit not partial_fit
            clf.fit(batch[:, :-1], batch[:, -1])
        else:
            clf.partial_fit(batch[:, :-1], batch[:, -1], classes=[0, 1])

        # Update progress
        print "\nBatch {0} finished. Time elapsed: {1}.\n\n\n".format(current_batch, str(datetime.now() - batch_start))
        current_batch += 1


    # Cross validation
    #predictions = clf.predict(cross_val_samples)

    #print classification_report(cross_val_sample_classes, predictions)

    print "Done, elapsed time: {}".format(str(datetime.now() - start))