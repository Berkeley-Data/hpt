

## (optional) GPU instance

Use `Deep Learning AMI (Ubuntu 18.04) Version 40.0` AMI
- on us-west-2, ami-084f81625fbc98fa4
- additional disk may be required for data 

Once logged in
```
# update conda to the latest 
conda update -n base conda 

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

```

## Installation

**Dependency repo**
- [modified OpenSelfSup](https://github.com/Berkeley-Data/OpenSelfSup)
- [modified SEN12MS](https://github.com/Berkeley-Data/SEN12MS) 
- [modified irrigation_detection](https://github.com/Berkeley-Data/irrigation_detection)

```bash
# clone dependency repo on the same levels as this repo and cd into this repo

# setup environment
conda create -n hpt python=3.7 ipython
conda activate hpt

# NOTE: if you are not using CUDA 10.2, you need to change the 10.2 in this command appropriately. Make sure to use torch 1.6.0
# (check CUDA version with e.g. `cat /usr/local/cuda/version.txt`)

# latest torch 
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# 1.6 torch (no support for torchvision transform on tensor)
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

#colorado machine 
conda install pytorch==1.2.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

# install local submodules
cd OpenSelfSup
pip install -v -e .
```

## Data installation

Installing and setting up all 16 datsets is a bit of work, so this tutorial shows how to install and setup RESISC-45, and provides links to repeat those steps with other datasets.

### RESISC-45
RESISC-45 contains 31,500 aerial images, covering 45 scene classes with 700 images in each class.

``` shell
# cd to the directory where you want the data, $DATA
wget -q https://bit.ly/3pfkHYp -O resisc45.tar.gz
md5sum resisc45.tar.gz  # this should be 964dafcfa2dff0402d0772514fb4540b
tar xf resisc45.tar.gz

mkdir ~/data 
mv resisc45 ~/data 

# replace/set $DATA and $CODE as appropriate 
# e.g., ln -s /home/ubuntu/data/resisc45 /home/ubuntu/OpenSelfSup/data/resisc45/all
ln -s $DATA/resisc45 $CODE/OpenSelfSup/data/resisc45/all

e.g., ln -s /home/ubuntu/data/resisc45 /home/ubuntu/hpt/OpenSelfSup/data/resisc45/all
```

### Download Pretrained Models
``` shell
mkdir OpenSelfSup/data/basetrain_chkpts
tools/download-pretrained-models.sh
```

## Verify Install With RESISC DataSet
[OpenSelfSup](https://github.com/Berkeley-Data/OpenSelfSup) 

Check installation by pretraining using mocov2, extracting the model weights, evaluating the representations, and then viewing the results on tensorboard or [wandb](https://wandb.ai/cal-capstone/hpt):


```bash
export WANDB_API_KEY=<use your API key>
export WANDB_ENTITY=cal-capstone
export WANDB_PROJECT=hpt3
#export WANDB_MODE=dryrun





cd OpenSelfSup

# Sanity check with single train and single epoch 
CUDA_VISIBLE_DEVICES=x ./tools/single_train.sh configs/selfsup/moco/r50_v2_resisc_in_basetrain_20ep.py --debug 

CUDA_VISIBLE_DEVICES=x ./tools/single_train.sh  configs/selfsup/moco/r50_v2_sen12ms_in_basetrain_20ep.py --work_dir work_dirs/selfsup/moco/r50_v2_sen12ms_in_basetrain_1ep/ --debug

# Sanity check: MoCo for 20 epoch on 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/selfsup/moco/r50_v2_resisc_in_basetrain_20ep.py 4

# if debugging, use 
tools/train.py configs/selfsup/moco/r50_v2_resisc_in_basetrain_1ep.py --work_dir work_dirs/selfsup/moco/r50_v2_resisc_in_basetrain_1ep/ --debug

# make some variables so its clear what's happening
CHECKPOINT=work_dirs/selfsup/moco/r50_v2_resisc_in_basetrain_20ep/epoch_20.pth
BACKBONE=work_dirs/selfsup/moco/r50_v2_resisc_in_basetrain_20ep/epoch_20_moco_in_basetrain.pth
# Extract the backbone
python tools/extract_backbone_weights.py ${BACKBONE} ${CHECKPOINT} 

# Evaluate the representations
./benchmarks/dist_train_linear.sh configs/benchmarks/linear_classification/resisc45/r50_last.py ${BACKBONE}

# View the results (optional if wandb is not configured)
cd work_dirs
# you may need to install tensorboard
tensorboard --logdir .
```


## setup sub-modules for sen12ms and openselfsup repo

Cloning
```console
git clone --recurse-submodules https://github.com/Berkeley-Data/hpt.git 

```

or alternatiely 
```
git submodule init
git submodule update
```

additional config 
```
git config push.recurseSubmodules on-demand
# show status including submodule 
git config status.submodulesummary 1
```

update
```
git submodule update --remote
```

For mroe info: [7.11 Git Tools - Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
 