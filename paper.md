## Abstract 




## Introduction 
The performance of deep convolutional neural networks depends on their capability and the amount of training data. The datasets are becoming larger in every domain and different kinds of network architectures like [VGG](https://arxiv.org/pdf/1409.1556.pdf), [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf), [ResNet](https://arxiv.org/pdf/1512.03385.pdf), [DenseNet](https://arxiv.org/pdf/1608.06993.pdf), etc., increased network models' capacity.  

However, the collection and annotation of large-scale datasets are time-consuming and expensive. Many self-supervised methods were proposed to learn visual features from large-scale unlabeled data without using any human annotations to avoid time-consuming and costly data annotations.  Contrastive learning of visual representations has emerged as the front-runner for self-supervision and has demonstrated superior performance on downstream tasks. All contrastive learning frameworks involve maximizing agreement between positive image pairs relative to negative/different images via a contrastive loss function; this pretraining paradigm forces the model to learn good representations. These approaches typically differ in how they generate positive and negative image pairs from unlabeled data and how the data are sampled during pretraining.  

Self-supervised approaches such as  Momentum  Contrast  (MoCo)  ([He et al.,  2019](https://arxiv.org/pdf/1911.05722.pdf);  [Chen et al.,2020](https://arxiv.org/pdf/2003.04297.pdf))  can leverage unlabeled data to produce pre-trained models for subsequent fine-tuning on labeled data.   In addition to MoCo,  these include frameworks such as SimCLR ([Chen et al., 2020](https://arxiv.org/pdf/2002.05709.pdf)) and PIRL ([Misra and Maaten, 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Misra_Self-Supervised_Learning_of_Pretext-Invariant_Representations_CVPR_2020_paper.pdf)).  

Remote sensing data has become broadly available at the petabyte scale, offering unprecedented visibility into natural and human activity across the Earth. In remote sensing, labeled data is usually scarce and hard to obtain. Due to the success of self-supervised learning methods, we explore their application to large-scale remote sensing datasets.  

While most self-supervised image analysis techniques focus on natural imagery, remote sensing differs in several critical ways. Natural imagery often has one subject; remote sensing images contain numerous objects such as buildings, trees, roads, rivers, etc. Additionally, the important content changes unpredictably within just a few pixels or between images at the same location from different times. Multiple satellites capture images of the same locations on earth with a wide variety of resolutions, spectral bands (channels), and revisit rates, such that any specific problem can require a different
combination of sensor inputs([Reiche et al., 2018](https://doi.org/10.1016/j.rse.2017.10.034),[Rustowicz et al., 2019](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.pdf)).  

While MoCo and other contrastive learning methods have demonstrated promising results on natural image classification tasks, their application to remote sensing applications has been limited.  

In this work, we demonstrate that pre-training [MoCo-v2](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) on data from multiple sensors lead to improved representations for remote sensing applications. 

## Related Work 

- SimCLR pre-training with BigEarthNet



## Problem Definition 

3.1. SEN12MS



3.3 


## Method 

 
#### Contrastive Learning Framework

**Temporal Positive Pairs**


**Resolution**


**Multiple-Sensor** 
different bands, different satelites 


**Dataset/metrics/evaluation **


## Experiments 







## Conclusion 



## References  
TODO: Use APA style later. Do this once the draft is ready by taking the links in the document, giving them a number and use APA style generator.  
[1] 
[2]
[3]
[4]
[5]

