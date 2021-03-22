# A. Question & Response
### 1. The evaluation options:
> a. Scene Classification -- land cover label (currently on wandb).

> b. Semantic segmentation -- assigning a class label to every pixel of the input image (not implement).

### 2. What metrics they used:
> a. Scene Classification -- Average Accuracy (single-label); Overall Accuarcy (multi-label); f-1, precision, and recall refer to the [repo](https://github.com/schmitt-muc/SEN12MS).

> b. Semantic segmentation -- class-wise and average accuracy -- refer to the [repo](https://github.com/lukasliebel/dfc2020_baseline).



# B. Summary of the Deep Dive

### 1. Scene Classification
1. Label were used -- **IGBP land cover scheme**.
a. the original IGBP land cover scheme has **17** classes.
b. the simplified version of IGBP classes has **10** classes, which derived and consolidated from the orignial 17 classes.

### 2. Semantic Segmentation





# Results (WIP on wandb, subject to changes)
> a. SEN12MS - Supervised Learning Benchmark - Classification

| Backbone  | Land Type  | Modalitities  | Bactch size  | Epochs | Accuracy (%) | Macro-F1 (%) | Micro-F1 (%) |
|---|---|---|---|---|---|---|---|
|DenseNet|single-label|_s1s2|64|100|51.16|50.78|62.90|
|DenseNet|single-label|_s2|64|100|54.41|52.32|64.74|
|ResNet50|single-label|_RGB|64|100|45.11|45.16|58.98|
|ResNet50|single-label|_s1s2|64|100|45.52|53.21|64.66|
|ResNet50|single-label|_s2|64|100|57.33|53.39|66.35|
|ResNet50|multi-label|_RGB|64|100|89.86|47.57|66.51|
|ResNet50|multi-label|_s1s2|64|100|91.22|57.46|71.40|
|ResNet50|multi-label|_s2|64|100|90.62|56.14|69.88|


