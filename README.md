# Hierarchical Pretraining: Research Repository

This is a research repository for the submission "Self-Supervised Pretraining Improves Self-Supervised Pretraining" 

For initial setup, refer to [setup instructions](references/setup.md). 

## Setup Weight & Biases Tracking 

```bash
export WANDB_API_KEY=<use your API key>
export WANDB_ENTITY=cal-capstone
export WANDB_PROJECT=hpt
#export WANDB_MODE=dryrun
```

## Base Training

[OpenSelfSup](https://github.com/Berkeley-Data/OpenSelfSup) 

Right now we assume ImageNet base trained models.
```bash
cd OpenSelfSup/data/basetrain_chkpts/
./download-pretrained-models.sh
```

## Pretraining With a New Dataset

[hpt](https://github.com/Berkeley-Data/hpt) 

We have a handy set of config generators to make pretraining with a new dataset easy and consistent!

**FIRST**, you will need the image pixel mean/std of your dataset, if you don't have it, you can do: 
```bash  
cd src/data/

# for sen12ms, run multiples times replacing --use_s1 by --use_s2 or --use_RGB
./compute-dataset-pixel-mean-std-sen12ms.py --data_dir /storage/sen12ms_x --data_index_dir /scratch/crguest/hpt/data --use_s1 --numworkers 1

# for others 
./compute-dataset-pixel-mean-std.py --data /scratch/crguest/data/sen12ms_small --numworkers 20 --batchsize 256

where image-folder has the structure from ImageFolder in pytorch
class/image-name.jp[e]g
or whatever image extension you're using
```
if your dataset is not arranged in this way, you can either:
(i) use symlinks to put it in this structure
(ii) update the above script to read in your data

NOTE: For sen12ms, the code is not working as expected (refer to [this issue](https://github.com/Berkeley-Data/hpt/issues/24), until then use the following. 
```
bands_mean = {'s1_mean': [-11.76858, -18.294598],
			  's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
						  2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}

bands_std = {'s1_std': [4.525339, 4.3586307],
			 's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
						1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]}
```

**NEXT**, copy the pretraining template
```bash
cd src/utils
cp templates/pretraining-config-template.sh pretrain-configs/sen12ms-small.sh
# edit pretrain-configs/sen12ms-small.sh

# once edited, generate the project
./gen-pretrain-project.sh pretrain-configs/my-dataset-config.sh
```

What just happened? We generated a bunch of pretraining configs in the following location (take a look at all of these files to get a feel for how this works):
```
OpenSelfSup/configs/hpt-pretrain/${shortname}
```

**NEXT**, you're ready to kick off a trial run to make sure the pretraining is working as expected =)

```bash
# the `-t` flag means `trial`: it'll only run a 50 iter pretraining
 ./utils/pretrain-runner.sh -t -d OpenSelfSup/configs/hpt-pretrain/${shortname}
```

**NEXT**, if this works, kick off the full training. NOTE: you can kick this off multiple times as long as the config directories share the same filesystem
```bash
# simply removing the `-t` flag from above
 ./utils/pretrain-runner.sh -d OpenSelfSup/configs/hpt-pretrain/${shortname}
```

**NEXT**, if you want to perform BYOL pretraining, add `-b` flag. 
```bash
# simply add the `-b` flag to above.
 ./utils/pretrain-runner.sh -d OpenSelfSup/configs/hpt-pretrain/${shortname} -b
```


Congratulations: you've launch a full hierarchical pretraining experiment.

**FAQs/PROBLEMS?**
* How does `pretrain-runner.sh` keep track of what's been pretrained?
    * In each config directory, it creates a `.pretrain-status` folder to keep track of what's processing/finished. See them with e.g.  `find OpenSelfSup/configs/hpt-pretrain -name '.pretrain-status'`
* How to redo a pretraining, e.g. because it crashed or something changed? Remove the
    * Remove the associate `.proc` or `.done` file. Find these e.g.
    ```bash
    find OpenSelfSup/configs/hpt-pretrain -name '.proc'
    find OpenSelfSup/configs/hpt-pretrain -name '.done'
    ```

## Evaluating Pretrained Representations
This has been simplified to simply:
```bash
./utils/pretrain-evaluator.sh -b OpenSelfSup/work_dirs/hpt-pretrain/${shortname}/ -d OpenSelfSup/configs/hpt-pretrain/${shortname}
```
where `-b` is the backbone directory and `-d` is the config directory. This command also works for cross-dataset evaluation (e.g. evaluate models trained on Resic45 and evaluate on UC Merced dataset).

**FAQ**

Where are the checkpoints and logs? E.g., if you pass in  `configs/hpt-pretrain/resisc` as the config directory,  then the working directories for this evalution is e.g. `work_dirs/hpt-pretrain/resisc/linear-eval/...`. If w&b is enabled, it will be logged on weight & biases 

## Finetuning
Assuming you generated the pretraining project as specified above, finetuning is as simple as:

```bash
./utils/finetune-runner.sh -d ./OpenSelfSup/configs/hpt-pretrain/${shortname}/finetune/ -b ./OpenSelfSup/work_dirs/hpt-pretrain/${shortname}/
```
where `-b` is the backbone directory and `-d` is the config directory
Note: to finetune using other backbones, simply pass in a different backbone directory (the script searches for `final_backbone.pth` files in the provided directory tree)


## Finetuning only on pretrained checkpoints with BEST linear analysis

First, specify the pretraining epochs which gives the best linear evaluation result in `./utils/top-linear-analysis-ckpts.txt`. Here is an example:

```
# dataset best-moco-bt best-sup-bt best-no-bt
chest_xray_kids 5000 10000 100000
resisc 5000 50000 100000
chexpert 50000 50000 400000
```
, in which for `chest_xray_kids` dataset, `5000`-iters, `10000`-iters, `100000`-iters are the best pretrained models under `moco base-training`, `imagenet-supervised base-training`, and `no base-training`, respectively.

Second, run the following command to perform finetuning only on the best checkpoints (same as above, except that the change of script name):
```bash
./utils/finetune-runner-top-only.sh -d ./OpenSelfSup/configs/hpt-pretrain/${shortname}/finetune/ -b ./OpenSelfSup/work_dirs/hpt-pretrain/${shortname}
```



## Pretraining on top of pretraining
Using the output of previously pretrained models, it is very easy to correctly setup pretraining on top of the pretraining.
Simply create a new config
```
utils/pretrain-configs/dataname1-dataname2.sh
```
(see `resisc-ucmerced.sh` for an example)

and then set the basetrained models to be the `final_backbone.pth` from the output of the last pretrained. e.g. for using resisc-45 outputs:

```
export basetrain_weights=(
    "work_dirs/hpt-pretrain/resisc/moco_v2_800ep_basetrain/50000-iters/final_backbone.pth"

    "work_dirs/hpt-pretrain/resisc/imagenet_r50_supervised_basetrain/50000-iters/final_backbone.pth"

    "work_dirs/hpt-pretrain/resisc/no_basetrain/200000-iters/final_backbone.pth"
)
```
(see `resisc-ucmerced.sh` for an example)

To select which backbones to use, evaluate the linear performance from the various source outputs (e.g. all the resisc pretrained outputs) on the target data (e.g. on uc-merced data). 

Then simply generate the project and execute the pretraining as normal:

```
./gen-pretrain-project.sh pretrain-configs/dataname1-dataname2.sh

./pretrain-runner.sh -d OpenSelfSup/configs/hpt-pretrain/$dataname1-dataname2
```


## Object Detection / Semantic Segmentation
Object detection/segmentation uses detectron2 and takes place in the directory
```
OpenSelfSup/benchmarks/detection
```

**First:** Check if the dataset configs you need are already present in `configs`. E.g. if you're working with CoCo, you'll see the following 2 configs:
```
configs/coco_R_50_C4_2x.yaml
configs/coco_R_50_C4_2x_moco.yaml
```
We'll use the config with the `_moco` suffix for all obj det and segmentation. If your configs already exist, skip the next step.

**Next:** assuming your configs do not exist, set up the configs you need for your dataset by copying an existing set of configs
```
cp configs/coco_R_50_C4_2x.yaml ${MYDATA}_R50_C4_2x.yaml
cp configs/coco_R_50_C4_2x_moco.yaml ${MYDATA}_R50_C4_2x_moco.yaml
```
Edit `${MYDATA}_R50_C4_2x.yaml` and set `MIN_SIZE_TRAIN` and `MIN_SIZE_TEST` to be appropriate for your dataset. Also, rename `TRAIN` and `TEST` to have your dataset name, set `MASK_ON` to `True` if doing semantic segmentation, and update `STEPS` and `MAX_ITER` if running the training for a different amount of time is appropriate (check relevant publications / codebases to set the training schedule).

Edit `${MYDATA}_R50_C4_2x_moco.yaml` and set `PIXEL_MEAN` and `PIXEL_STD` (use `compute-dataset-pixel-mean-std.py` script above, if you don't know them).

Then, edit `train_net.py` and add the appropriate data registry lines for your train/val data
```
register_coco_instances("dataname_train", {}, "obj-labels-in-coco-format_train.json", "datasets/dataname/dataname_train")
register_coco_instances("dataname_val", {}, "obj-labels-in-coco-format_val.json", "datasets/dataname/dataname_val")
```

Then, setup symlinks to your data under `datasets/dataname/dataname_train` and `datasets/dataname/dataname_val`, where you replace dataname with your dataname used in the config/registry.

**Next**, convert your backbone(s) to detectron format, e.g. (NOTE: I recommend keeping backbones in the same directory that they are originally present in, and appending a `-detectron2` suffix)
```
python convert-pretrain-to-detectron2.py ../../data/basetrain_chkpts/imagenet_r50_supervised.pth ../../data/basetrain_chkpts/imagenet_r50_supervised-detectron2.pth
```

**Next** kick off training
```
python train_net.py --config-file configs/DATANAME_R_50_C4_24k_moco.yaml --num-gpus 4 OUTPUT_DIR results/${UNIQUE_DATANAME_EXACTLY_DESCRIBING_THIS_RUN}/ TEST.EVAL_PERIOD 2000 MODEL.WEIGHTS ../../data/basetrain_chkpts/imagenet_r50_supervised-detectron2.pth SOLVER.CHECKPOINT_PERIOD ${INT_HOW_OFTEN_TO_CHECKPOINT}
```
results will be in `results/${UNIQUE_DATANAME_EXACTLY_DESCRIBING_THIS_RUN}`, and you can use tensorboard to view them.

## Commit and Share Results
Run the following command to grab all results (linear analysis, finetunes, etc) and put them into the appropriate json results file in `results/`:
```
./utils/update-all-results.sh
```

You can verify the results in `results` and then add the new/updated results file to git and commit.

**Did you get an error message such as:**
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Please investigate as your results may not be complete.
(see errors in file: base-training/utils/tmp/errors.txt)

will not include partial result for /home/XXX/base-training/utils/../OpenSelfSup/work_dirs/hpt-pretrain/resisc/finetune/1000-labels/imagenet_r50_supervised_basetrain/50000-iters-2500-iter-0_01-lr-finetune/20200911_170916.log.json
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```
This means that this particular evaluation run did not appear to run for enough iterations. Investigate the provided log file, rerun any necessary evaluations, and remove the offending log file. 

**Debugging this script** this script finds the top val accuracy, and save the corresponding test accuracy using the following script:
```
./utils/agg-results.sh
```
which outputs results to `utils/tmp/results.txt` and errors to `utils/tmp/errors.txt`. Look at this file if your results aren't being generated correctly.

## Generate plots

```bash
cd utils
python plot-results.py
```

See plots in directory `plot-results`
(you can also pass in a `--data` flag to only generate plots for a specific dataset, e.g. `python plot-results.py --data resisc`)


**To plot the eval & test acc curves**, use `./utils/plot.py`
```bash
cd utils
python plot.py --fname PLOT_NAME --folder FOLDER_CONTAINING_DIFFERENT_.PTH_FOLDERs
```

**To Generate plot for Exp-2-finetuning**, do
```bash
bash utils/plot-results-exp-2.sh
```

See plot in directory `plot-results/exp-2`.

**To Generate plot for Exp-3-Hierarchical Pretraining**, do
```bash
bash utils/plot-results-exp-3.sh
```

See plot in directory `plot-results/exp-3`.


## Getting activations for similarity measures

Run `get_acts.py` with a model used for a classifaction task
(one that has a test/val set).\
Alternatively, run dist_get_acts as follows:
```shell
bash dist_get_acts.sh ${CFG} ${CHECKPOINT} [--grab_conv...]
```
Default behavior is to grab the entire batch of linear layers.
Setting `--grab_conv` will capture a single batch of all convolutional layers.\
Layers will be saved in `${WORK_DIR}/model_acts.npz`.
The npz contains a dictionary which maps layer names to the activations.


## Debugging and Developing Within OpenSelfSup

Here's a command that will allow breakpoints (WARNING: the results with the debug=true flag SHOULD NOT BE USED -- they disable sync batch norms and are not comparable to other results):

```bash
# from OpenSelfSup/
# replace with your desired config
python tools/train.py configs/hpt-pretrain/resisc/moco_v2_800ep_basetrain/500-iters.py --work_dir work_dirs/debug --debug

```

## (Other) BigEarthNet bands mean and standard deviation

For S-1 data, band name {'VV', 'VH'}
```bash
python dataset_band_info_BigEarthNet.py\
 --path <data_dir>\
 --data_index_dir <data index dir>\
 --numworkers 30\
 --batchsize 256\
 --use_s1\
 --band <band name>

```






