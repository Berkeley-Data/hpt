# A. Question & Response
### 1. The evaluation options:
> a. Scene Classification -- land cover label (currently on wandb).

> b. Semantic segmentation -- assigning a class label to every pixel of the input image (not implement).

### 2. What metrics they used:
> a. Scene Classification -- Average Accuracy (single-label); Overall Accuarcy (multi-label); f-1, precision, and recall refer to the [repo](https://github.com/schmitt-muc/SEN12MS).

> b. Semantic segmentation -- class-wise and average accuracy -- refer to the [repo](https://github.com/lukasliebel/dfc2020_baseline).

**Recalling from the meeting with Colorado, whether this metircs are standard? -- the answer is yes -- hence, In stead of using the author's evaluation system, there maybe options for the use of openselfsup ecosystem.**


# B. Summary of the Benchmark Evaluation Deep Dive

### 1. Scene Classification
1. Label were used -- **IGBP land cover scheme**.

> a. the original IGBP land cover scheme has **17** classes.

> b. the simplified version of IGBP classes has **10** classes, which derived and consolidated from the orignial 17 classes.

2. Definition of single-label and multi-label.

> a. For every scence (patch), we can identify the labels through land cover images from MODIS, in which the first band describes the IGBP classification scheme, whereas the rest of the three bands covered the LCCS land cover layer, LCCS land use layer, and the LCCS surface hydrology layer. According to the authors, the overall acc for the layers are about 67% (IGBP), 74% (LCCS land cover), 81% (LCCS land use), and 87% (LCCS surface hydrology). There are known label noise to the SEN12MS dataset and hence these accuracies will constitute the upper bound of actually achievable predictive power.

> b. from (a), the authors has already processed and stored the labels of each image in SEN12MS with full IGBP classes into the file **IGBP_probability_labels.pkl**, meaning the percentage of the imange that belongs to each classes, where further label types and target classes can be derived during the training steps -- single label or multi-label for a scence (patch). Below is the parameters we can define on the fly when training. 

>> - full classes (17) or simplified classes (10)
>> -  single label -- it's derived from the probabilities files that applys the argmax to select the highest probability of class (vector) in a scence (patch).
>> - multi label -- it's derived from the probabilities files that some threshold can be applied for each classes in a vetor.

> c. For the single-label, the authors also provided the processed one-hot encoding for the vector dervided from (b).

>> - single-label_IGBPfull_ClsNum: This file contains scene labels based on the full IGBP land cover scheme, represented by actual class numbers.
>> - single-label_IGBP_full_OneHot: This file contains scene labels based on the full IGBP land cover scheme, represented by a one-hot vector encoding.
>> - single-label_IGBPsimple_ClsNum: This file contains scene labels based on the simplified IGBP land cover scheme, represented by actual class numbers.
>> - single-label_IGBPsimple_OneHot: This file contains scene labels based on the simplified IGBP land cover scheme, represented by a one-hot vector encoding. All these files are available both in plain ASCII (.txt) format, as well as .pkl format.

3. Modalities
The modalities can be chosen when performing the training. Three options can be evaluated. 
>> - _RGB: only S2 TGB imagery is used
>> _s2: full multi-spectral s-2 data were used
>> _s1s2: data fusion-based models analyzing both s-1 and s-2 data

**Checked whether _s1s2 would be the most releveant model when it comes to compares with our approach - s1s2 MOCO, or it does not matter?**

4. Reporting Metrics
The authors has implemented some metrics in the .py files but according to the papers, there is no actual reporting for the model describe above (**or not found, still searhing**). However, the author did mentioned in the paper as well as in the .py files for the metrics to be reported, which includes:
>> 1. Average Accuracy (get_AA) -- only applied to single-label types.
>> 2. Overall Accuracy (OA_multi) -- particular for multi-label cases.
>> 3. F1-score, precision, and recall -- this is relatively standard measure.

5. There are pre-trained model(weights) and optimizations parameters can be downloaded.


### 2. Semantic Segmentation (WIP)
-- this tasks seems to be not straightforwrad. and the author did not report everything (based on the paper and repo). checking ...

WIP



# C. Our Evaluation choices

### a. methods

1. potential 1 -- using the exiting scence classificaion models and the current evalution in sen12ms dataset to evaluate the moco one
2. potential 2 -- using openselfsup to evalute the sen12ms dataset?? (tbc)
3. potential 3 -- others ?? (tbd)

### b. samples
1. full or sub-samples ? (distributions)
2. size


# D. Results (WIP on wandb, subject to changes)
### 1. SEN12MS - Supervised Learning Benchmark - Scence Classification
These models were downloaded from their per-trained described in B-5, and evaluated.

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


# E. Appendix
1. IGBP Land Cover Classification System
![Screen Shot 2021-03-21 at 10 52 56 PM](https://user-images.githubusercontent.com/39634122/111934636-2f68ee00-8a98-11eb-8763-8453266227ed.png)






