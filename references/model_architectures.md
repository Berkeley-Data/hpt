#### Key model architectures and terms:
-   Supervised training (full dataset)
    -   baseline: downloaded the pre-trained the models and evaluate without finetuning.
-   Supervised training (1k dataset)
    -   Supervised: original ResNet50 used by Sen12ms
    -   Supervised_1x1: adding conv1x1 block to the ResNet50 used by Sen12ms
-   Finetune/transfer learning (1k dataset)
    -   Moco: the ResNet50 used by Sen12ms is initialized with the weight from Moco backbone
    -   Moco_1x1: adding conv1x1 block to the ResNet50 used by Sen12ms and both input module and ResNet50 layers are initialized with the weight from Moco
    -   Moco_1x1Rnd: adding conv1x1 block to the ResNet50 used by Sen12ms. ResNet50 layers are initialized with the weight from Moco but input module is initialized with random weights
-   Finetune v2 (1k dataset)
    -   freezing ResNet50 fully or partially does not seem to help with accuracy. We will continue explore and share the results once we are sure there is no issue with implementation.


#### Key pretrained models 

Some pretrained models: 

**Sensor Augmentation** 
- [vivid-resonance-73](https://wandb.ai/cjrd/BDOpenSelfSup-tools/runs/3qjvxo2p)
- [silvery-oath-7](https://wandb.ai/cal-capstone/hpt2/runs/2rr3864e) 
- sen12_crossaugment_epoch_1000.pth: 1000 epocs 

**Data Fusion - Augmentation Set 2**
- [(optional fusion) crimson-pyramid-70](https://wandb.ai/cal-capstone/hpt4/runs/2iu8yfs6): 200 epochs 
- [(partial fusion) laced-water-61](https://wandb.ai/cal-capstone/hpt4/runs/367tz8vs) 200 epochs, 32K
- [(partial fusion) visionary-lake-62](https://wandb.ai/cal-capstone/hpt4/runs/1srlc7jr/overview?workspace=user-taeil)  should deprecate. different number of epochs from other pretrained models 
- [(full fusion) electric-mountain-33](https://wandb.ai/cal-capstone/hpt4/runs/ak0xdbfu)

**Data Fusion - Augmentation Set 1**