__author__ = 'acpigeon'

import pandas as pd
import numpy as np
import random
from datetime import datetime


def get_data(train=True):
    """Read data in from the file and yield row by row"""

    if train:
        path = '../data/train_rev1_short_short.csv'  # 10 rows
        #path = '../data/train_rev1_short.csv'  # 1000 rows
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
                    x[ixx - 2] = abs(hash(str(ixx) + '_' + f)) % (2 ** 4)  # offset to account for removing id and class

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
    """Write 20% of the sample to a file to retrieve later for cross validation. This lets us
    continue to avoid keeping the main training set in memory."""
    for ids, row, classes in get_data():
        # print row
        this_row = np.zeros(cols + 1)
        for i in row:
            this_row[i] = 1
        sorting_hat = random.choice([1, 2, 3, 4, 5])  # pick 1-5 randomly
        if sorting_hat == 5:  # every fifth element goes to xval
            yield (0, this_row)
        else:
            yield (1, this_row)


if __name__ == '__main__':
    start = datetime.now()

    # record this value so we don't have to get it every run on the large trains...
    columns = get_len_feature_array()
    #cols = 1048575  # full train set, we can cheat on the full run and save about 18 minutes


    for x, y in train_test_split(columns):
        if x:
            print "train: " + str(y)
        else:
            print "xval:" + str(y)

    #print get_len_feature_array()
    print('Done, elapsed time: %s' % str(datetime.now() - start))