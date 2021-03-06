#!/bin/bash

# COPY THIS SCRIPT AND SET ALL OF THE VARIABLES


###############
# PRETRAINING #
###############

# shortname: the short name used to reference the dataset
export shortname="coco_2014"

# the RGB pixel means/std of the dataset,
# DON'T just use the default if you don't know it!
# Compute using: ./compute-dataset-pixel-mean-std.py
export pixel_means="0.4702, 0.4470, 0.4076"
export pixel_stds="0.2785, 0.2740, 0.2889"

# how many iterations of pretraining? (each iter is one minibatch of 256)
# with basetraining
export bt_iters="50,500,5000,50000"
# without basetraining
export no_bt_iters="5000,50000,100000,200000"

# We pretrain on both training and validation sets
# BUT NOT TEST!!!
# This file should have a list of all images for pretraining
# i.e. the train and val set
# it should NOT have labels (you'll get an error if it does)
export train_val_combined_list_path="/rscratch/data/coco_2014/images/train_val.txt"

# the list of images just for training (used for the linear evaluation)
# NEEDS labels
export train_list_path="/rscratch/data/coco_2014/images/train.txt"
# the list of images just for training (used for val on the linear evaluation)
# NEEDS labels
export val_list_path="/rscratch/data/coco_2014/images/val.txt" 
# the list of images just for training (used for test on the linear evaluation)
# NEEDS labels
export test_list_path="/rscratch/data/coco_2014/images/test.txt" 

# the base data path that the image lists reference
export base_data_path="/rscratch/data/coco_2014/images"

#
# OPTIONAL - only change if you know what you're doing ;)
#

# 224 is a standard, but may not be appropriate for you data
export crop_size="224"

# basetrain weights, update this array with experiments you're running
export basetrain_weights=(
        # standard moco basetrain
        "data/basetrain_chkpts/moco_v2_800ep.pth"

        # supervised basetrain
        "data/basetrain_chkpts/imagenet_r50_supervised.pth"

        # no basetrain
        ""
)

########
# Eval #
########

# COMMENT OUT THIS SECTION IF YOU DO NOT WANT TO GEN EVAL CONFIGS

# assuming the linear/semi-sup is a image classification
# (need to create your own config for other eval problems)
export num_classes="80" # eg 10 for cifar-10

# resize images to this size before taking a center crop of crop_size (defined above)
export test_precrop_size="256"

## NOTE: only change these if you know what you're doing
export linear_iters="5000"

export linear_lr_drop_iters="1651,3333" # when we drop the LR by 10 (1/3 and 2/3)
export linear_lr='30' # linear layer learning rate

# number of times to run the linear eval
export linear_reruns=3


############
# Finetune #
############

export ft_num_train_labels="1000,all" # do 1000 label and all train finetune evaluation
# TODO(cjrd) add number of finetune reruns (use different datasets of 100 and 1k labels)

# learning rates
export ft_lrs="0.01,0.001"

# finetuning amount when done by epochs
export ft_by_epoch="90"
export ft_by_epoch_lr_steps="30,60"

# finetuning amount when done by iters
export ft_by_iter="2500"
export ft_by_iter_lr_steps="833,1667"

##########
# Extras #
##########

# you may need to reduce this number if your cpu load is too high
export workers_per_gpu=4

# Uncomment if using multiclass problem
export dataset_type="AUROCDataset"
export image_head_class_type="ImageListMultihead"
export bce_string="use_bce_loss=True"
# map the input values to classes (see chexpert.sh)
export class_map=""
export eval_params="dict()"
