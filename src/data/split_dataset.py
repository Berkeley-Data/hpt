import random
import math
import pickle as pkl
from glob import glob
import argparse
import math
import os
import csv
import re

# Enhance it to take arguments
# It splits and writes to pkl file
def split_data():
    # Configure paths to your dataset files here
    DATASET_FILE = 'data.txt'
    FILE_TRAIN = 'train_list_updated.txt'
    FILE_VALID = 'val_list_updated.txt'
    FILE_TESTS = 'test_list_updated.txt'

    # Set to true if you want to copy first line from main
    # file into each split (like CSV header)
    IS_CSV = True

    # Make sure it adds to 100, no error checking below
    PERCENT_TRAIN = 88.6
    PERCENT_VALID = 11.4
    PERCENT_TESTS = 0

    data = [l.rstrip('\n') for l in open(DATASET_FILE, 'r')]

    train_file = open(FILE_TRAIN, 'w')
    valid_file = open(FILE_VALID, 'w')
    tests_file = open(FILE_TESTS, 'w')

    if IS_CSV:
        train_file.write(data[0])
        valid_file.write(data[0])
        tests_file.write(data[0])
        data = data[1:len(data)]

    num_of_data = len(data)
    num_train = int((PERCENT_TRAIN / 100.0) * num_of_data)
    num_valid = int((PERCENT_VALID / 100.0) * num_of_data)
    num_tests = int((PERCENT_TESTS / 100.0) * num_of_data)

    data_fractions = [num_train, num_valid, num_tests]
    split_data = [[], [], []]

    rand_data_ind = 0

    for split_ind, fraction in enumerate(data_fractions):
        for i in range(fraction):
            rand_data_ind = random.randint(0, len(data) - 1)
            split_data[split_ind].append(data[rand_data_ind])
            data.pop(rand_data_ind)

    for l in split_data[0]:
        train_file.write(l)

    for l in split_data[1]:
        valid_file.write(l)

    for l in split_data[2]:
        tests_file.write(l)

    train_file.close()
    valid_file.close()
    tests_file.close()

    # Write picke files
    pickle_out = open("train_list_updated.pkl", "wb")
    pkl.dump(split_data[0], pickle_out)
    pickle_out.close()

    pickle_out = open("val_list_updated.pkl", "wb")
    pkl.dump(split_data[1], pickle_out)
    pickle_out.close()

    pickle_out = open("test_list_updated.pkl", "wb")
    pkl.dump(split_data[2], pickle_out)
    pickle_out.close()

    # Test by reading back
    sample_list = pkl.load(open("val_list_updated.pkl", "rb"))
    print("Iterating")
    for list_entry in sample_list:
        print(list_entry)

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(
        description='This script provides multiple util functions which are used for data creation/processing.')
    parser.add_argument('-sd', '--split_data', default=False, type=str2bool,
                        help="whether to split the data")

    args = parser.parse_args()

    if args.split_data:
        print('split_data---START')
        print(split_data())
        print('split_data---END')
