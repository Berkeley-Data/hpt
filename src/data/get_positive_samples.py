import random
import math
import pickle as pkl
from glob import glob
import argparse
import math
import os
import csv
import re

def multisplit(s, delims):
    pos = 0
    for i, c in enumerate(s):
        if c in delims:
            yield s[pos:i]
            pos = i + 1
    yield s[pos:]

# Returns a dictionary of positive pairs
# Sample (positive pair key,value): spring_100_p101 : ['ROIs1158_spring_lc_100_p101.tif', 'ROIs1158_spring_s1_100_p101.tif', 'ROIs1158_spring_s2_100_p101.tif']
# Run command: python utils.py -gpp True -gppf train_list.pkl
def get_positive_samples(file_to_parse):
    """
    The following is a positive pair:
    ROIs1158_spring_lc_100_p101.tif
    ROIs1158_spring_s1_100_p101.tif
    ROIs1158_spring_s2_100_p101.tif
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

        key = file_name_splits[1]+'_'+file_name_splits[3]+'_'+file_name_splits[4]
        if key in positive_pairs:
            list_val = positive_pairs[key]
            list_val.append(file_name)
            #print('list_val', list_val)
            positive_pairs[key] = list_val
        else:
            positive_pairs[key] = [file_name]

    for positive_key in positive_pairs:
        val_list = positive_pairs[positive_key]
        if len(val_list) > 1:
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

    # get positive pairs
    parser.add_argument('-gps', '--get_positive_samples', default=False, type=str2bool,
                        help="whether get positive samples in SEN12MS data file")
    parser.add_argument('-gpsf', '--positive_sample_file', type=str,
                        help="file to be parsed")

    args = parser.parse_args()

    if args.get_positive_samples:
        print('get_positive_samples---START')
        get_positive_samples(file_to_parse=args.positive_sample_file)
        print('get_positive_samples---END')
