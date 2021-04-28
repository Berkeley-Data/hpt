
#### Fusion approach 

**SEN12MS (1024)**
| aug set 2| s1 | s2 | s1/s2 | Note | 
| --- | --- | --- | --- | --- | 
| Supervised () | ? | ?| ? |  | 
| [all fusion]() | ? | ? | ? | running |
| [partial fusion]() | ? | ? | ? | done |
| [optional fusion]() | ? | ? | ? | done |

**SEN12MS (512)**
| aug set 2| s1 | s2 | s1/s2 | Note | 
| --- | --- | --- | --- | --- | 
| Supervised () | ? | ?| ? |  | 
| [all fusion]() | ? | ? | ? | running |
| [partial fusion]() | ? | ? | ? | done |
| [optional fusion]() | ? | ? | ? | done |

**BigEarthNet**
| aug set 2| s1 | s2 | s1/s2 | Note | 
| --- | --- | --- | --- | --- | 
| Supervised (1024) | ? | ?| ? | running | 
| [all fusion]() | ? | ? | ? | running |
| [partial fusion]() | ? | ? | ? | done |
| [optional fusion]() | ? | ? | ? | done |

**BigEarthNet (512)**
| aug set 2| s1 | s2 | s1/s2 | Note | 
| --- | --- | --- | --- | --- | 
| Supervised (1024) | ? | ?| ? | running | 
| [all fusion]() | ? | ? | ? | running |
| [partial fusion]() | ? | ? | ? | done |
| [optional fusion]() | ? | ? | ? | done |


#### sensor augmentation 

| | Metrics|single-label |multi-label | Note | 
| --- | --- | --- | --- | --- | 
|  |  |   |   | | 
| full dataset | Supervised s2	|  .57	| .60| |
| | Supervised s1/s2	| .45	| .64|| 
| | Supervised RGB | .45	| .58| |
| |  |   |   | | 
|s2 | Supervised 1x1	| .3863	| .4893 | | 
|  | Supervised	| .4355	| .5931 | too good?| 
|  | Moco 1x1 RND | .4345 | .6004 | 	| 
|  | Moco 1x1 | .4469	| **.601**| not necessarily better | 
|  | Moco 1x1 RND (1000ep) | .4264 | .5757 | overfitting? | 
|  | Moco 1x1 (1000ep) | .4073	| .5622 | overfitting? | 

| | Metrics|single-label |multi-label | Note | 
| --- | --- | --- | --- | --- | 
|s1/s2 | :white_check_mark: Supervised 1x1 | .4094	| .5843 | | 
|  | :white_check_mark: Supervised	| .4426	| .4678 | | 
|  | :no_entry_sign: Moco 1x1 RND | .4477 | .5317 | | 
| | :no_entry_sign: Moco 1x1 |  .4474	| .5302 | no conv1 weight transfer | 
|  | :no_entry_sign: **Moco**	| .4718	| **.6697** | no conv1 weight transfer |

- single-label: Average Accuracy 
- multi-label: Overall Accuracy 


crimson-pyramid 

**aug set 1(TBD)**

| aug set 1| s1 | s2 | s1/s2 | Note | 
| --- | --- | --- | --- | --- | 
| Supervised (full) | xx | xx | xx | xx | 
| Supervised (1024) | xx | xx | xx | xx |
| --- | --- | --- | --- | --- | 
| [sensor-based augmentation] | xx | xx | xx | xx | 
| [all fusion] | xx | xx|  xx | xx |
| [partial fusion] | xx | xx | xx | xx |
| [optional fusion] | xx | xx | xx | xx|


**aug set 2**

| aug set 2| s1 | s2 | s1/s2 | Note | 
| --- | --- | --- | --- | --- | 
| Supervised (full) | [Pretrained model is not provided](https://syncandshare.lrz.de/getlink/fiCDbqiiSFSNwot5exvUcW1y/trained_models) | [.60](https://wandb.ai/cal-capstone/sup_scene_cls/runs/3mg9zr5t) | [.64](https://wandb.ai/cal-capstone/sup_scene_cls/runs/2lda2016) | need to retest s1, s2 with zero padding | 
| Supervised (1024) | [0.4003](https://wandb.ai/cal-capstone/sup_scene_cls/runs/555fv4cb) | [0.6108](https://wandb.ai/cal-capstone/sup_scene_cls/runs/3m1h27zt) | [.5856](https://wandb.ai/cal-capstone/sup_scene_cls/runs/dpwjby4o) | | 
| --- | --- | --- | --- | --- | 
| [sensor-based augmentation] | - | [0.6277](https://wandb.ai/cal-capstone/SEN12MS/runs/2826nuca) | [0.6697](https://wandb.ai/cal-capstone/SEN12MS/runs/22tv0kud) | xx | 
| [all fusion](https://wandb.ai/cal-capstone/hpt4/runs/ak0xdbfu/overview) | xx | [.6251]? | [.5957](https://wandb.ai/cal-capstone/scene_classification/runs/2y2q8boi) | |
| [partial fusion](https://wandb.ai/cal-capstone/hpt4/runs/367tz8vs) | [.4729](https://wandb.ai/cal-capstone/scene_classification/runs/1qx384cs) | [.5812](https://wandb.ai/cal-capstone/scene_classification/runs/1bdmms2d) |[.6072](https://wandb.ai/cal-capstone/scene_classification/runs/1meu9iym) | |
| [optional fusion](https://wandb.ai/cal-capstone/hpt4/runs/2iu8yfs6) | [.4824](https://wandb.ai/cal-capstone/scene_classification/runs/tu3vuefx) | [.5601](https://wandb.ai/cal-capstone/scene_classification/runs/2hdbuxtv) | [.5884](https://wandb.ai/cal-capstone/scene_classification/runs/y5x2xce6) | |


- Supervised (full) s1, s2 need to be retested with zero padding 12 channel. 


#### BigEarthNet Evaluation (TBD)
scence classification (multi or single label?)

**aug set 1(TBD)**


| aug set 1| s1 | s2 | s1/s2 | Note | 
| --- | --- | --- | --- | --- | 
| Supervised (full) | xx | xx | xx | xx | 
| Supervised (1024) | xx | xx | xx | xx |
| --- | --- | --- | --- | --- | 
| [sensor-based augmentation] | xx | xx | xx | xx | 
| [all fusion]  | xx | xx | xx | xx |
| [partial fusion] | xx | xx | xx | xx |
| [optional fusion] | xx | xx | xx | xx|


**aug set 2**

| aug set 2| s1 | s2 | s1/s2 | Note | 
| --- | --- | --- | --- | --- | 
| Supervised (full) | xx | xx | xx | xx | 
| Supervised (1024) | [.4008](https://wandb.ai/cal-capstone/sup_scene_cls/runs/1lnfsmdi) | [.5496](https://wandb.ai/cal-capstone/sup_scene_cls/runs/3fpzht5f) | [.5423](https://wandb.ai/cal-capstone/sup_scene_cls/runs/1qma48o1) | xx |
| --- | --- | --- | --- | --- | 
| [sensor-based augmentation] | xx | xx | xx | xx | 
| [all fusion]  | xx | xx | xx | xx |
| [partial fusion] | [.4279](https://wandb.ai/cal-capstone/scene_classification/runs/2a1tlnbv) | [.5351](https://wandb.ai/cal-capstone/scene_classification/runs/2f0pjxwx) | [.5352](https://wandb.ai/cal-capstone/scene_classification/table?workspace=user-kenhan) | xx |
| [optional fusion] | [.4478](https://wandb.ai/cal-capstone/scene_classification/runs/36c8z6ae) | [.5120](https://wandb.ai/cal-capstone/scene_classification/runs/3oazvjke) | [.5294](https://wandb.ai/cal-capstone/scene_classification/runs/nar53xcn) | xx|

- Supervised (full) s1, s2 need to be retested with zero padding 12 channel. 
