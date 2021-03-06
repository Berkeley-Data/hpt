#### Primary datasets

- RESISC-45: contains 31,500 aerial images, covering 45 scene classes with 700 images in each class.
- SEN12MS: triplet. refer to [SEN12MS EDA notebook](notebooks/eda_sen12ms.ipynb) 


### SEN12MS 

```bash 

# create softlink to SEN12MS datasets on different storage or different path 

mkdir /scratch/crguest/OpenSelfSup/data/sen12ms
ln -s /storage/sen12ms_x /scratch/crguest/OpenSelfSup/data/sen12ms/all

```

To prepare smaller set (1 patch from one region), refer to [this notebook](https://github.com/Berkeley-Data/hpt/blob/master/notebooks/1.1-tg-prepare-subset-sen12ms.ipynb)  

### Additional datasets:
Following the same instructions as above except replace the bit.ly 
* BDD: https://bdd-data.berkeley.edu
* xView: http://xviewdataset.org
* UC-Merced: http://weegee.vision.ucmerced.edu/datasets/landuse.html
* DomainNet: http://ai.bu.edu/DomainNet/
* Chexpert: https://stanfordmlgroup.github.io/competitions/chexpert/
* Chest-X-ray-kids:https://www.kaggle.com/andrewmvd/pediatric-pneumonia-chest-xray
* Flowers: https://www.robots.ox.ac.uk/~vgg/data/flowers/
* VIPER: https://playing-for-benchmarks.org
* CoCo: https://cocodataset.org/#home
* PASCAL: http://host.robots.ox.ac.uk/pascal/VOC/