## download pre-trained models 

Some of key pre-trained models are on s3 (s3://sen12ms/pretrained): 
- [200 epochs w/o augmentation: vivid-resonance-73](https://wandb.ai/cjrd/BDOpenSelfSup-tools/runs/3qjvxo2p/overview?workspace=user-cjrd) 
- [20 epochs w/o augmentation: silvery-oath7-2rr3864e](https://wandb.ai/cal-capstone/hpt2/runs/2rr3864e?workspace=user-taeil) 
- [sen12ms-baseline: soft-snowflake-3.pth](https://wandb.ai/cal-capstone/SEN12MS/runs/3gjhe4ff/overview?workspace=user-taeil)

```
aws configure 
aws s3 sync s3://sen12ms/pretrained . --dryrun
aws s3 sync s3://sen12ms/pretrained_sup . --dryrun
```

Any other models can be restored by run ID if stored with W&B. Go to files section under the run to find `*.pth` files  

#### Extract and Convert Backbone 
```
# method 1: from working dir
CHECKPOINT=work_dirs/selfsup/moco/r50_v2_resisc_in_basetrain_20ep/epoch_20.pth
# method 2: from W&B, {projectid}/{W&B run id}
CHECKPOINT=hpt3/2brjqb28

# Extract the backbone
python classification/models/convert_moco_to_resnet50.py -i hpt3/2brjqb28 -o pretrained/moco
```

#### Evaluate the representations :confused: :question:

```
./benchmarks/dist_train_linear.sh configs/benchmarks/linear_classification/resisc45/r50_last.py ${BACKBONE}
``` 