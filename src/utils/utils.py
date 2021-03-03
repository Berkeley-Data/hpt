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

def multisplit(s, delims):
    pos = 0
    for i, c in enumerate(s):
        if c in delims:
            yield s[pos:i]
            pos = i + 1
    yield s[pos:]

# Returns a dictionary of positive pairs
# Sample (positive pair key,value): 81_p99 : ['ROIs1868_summer_s2_81_p99.tif', 'ROIs1970_fall_s2_81_p99.tif', 'ROIs2017_winter_s2_81_p99.tif']
# Run command: python utils.py -gpp True -gppf train_list.pkl
def get_positive_pairs(file_to_parse):
    """
    The following is a positive pair:
    ROIs1158_spring_lc_100_p101.tif
    ROIs1158_spring_s1_100_p101.tif
    ROIs1158_spring_s2_100_p101.tif
    ROIs1868_summer_lc_100_p101.tif
    ROIs1868_summer_s2_100_p101.tif
    ROIs1970_fall_s1_100_p101.tif
    """
    if not os.path.exists(file_to_parse):
        print('ERROR: file', file_to_parse, 'does not exist')

    splits = []
    if(file_to_parse.split('.')[1]== 'pkl'):
        splits = pkl.load(open(file_to_parse, "rb"))
    else:
        # reading csv file
        with open(file_to_parse, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
            # extracting each data row one by one
            for row in csvreader:
                splits.append(row[0])
                # get total number of rows
            print("Total no. of rows: %d" % (csvreader.line_num))

    positive_pairs = {}
    for file_name in splits:
        file_name = file_name.rstrip()
        #print(file_name)S
        file_name_splits = list(multisplit (file_name, '_.'))
        #print(file_name_splits)

        key = file_name_splits[3]+'_'+file_name_splits[4]
        if key in positive_pairs:
            list_val = positive_pairs[key]
            list_val.append(file_name)
            #print('list_val', list_val)
            positive_pairs[key] = list_val
        else:
            positive_pairs[key] = [file_name]

    for positive_key in positive_pairs:
        val_list = positive_pairs[positive_key]
        if len(val_list) > 2:
            print(positive_key, ':', val_list)

    return positive_pairs


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
    # get positive pairs
    parser.add_argument('-gpp', '--get_positive_pairs', default=False, type=str2bool,
                        help="whether get positive pairs in SEN12MS data file")
    parser.add_argument('-gppf', '--positive_pair_file', type=str,
                        help="file to be parsed")

    args = parser.parse_args()

    if args.split_data:
        print('split_data---START')
        print(split_data())
        print('split_data---END')

    if args.get_positive_pairs:
        print('get_positive_pairs---START')
        get_positive_pairs(file_to_parse=args.positive_pair_file)
        print('get_positive_pairs---END')
