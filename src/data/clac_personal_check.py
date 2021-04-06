
######
import glob
import os
s1 = []
for file in [os.path.basename(x) for x in glob.glob("/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0/*")]:
    s1.append(file)

###
s1_vh = []
for file in glob.glob("/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0/*"):
    for img in glob.glob(file+'/*_VH.tif'):
        s1_vh.append(img)

s1_vv = []
for file in glob.glob("/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0/*"):
    for img in glob.glob(file+'/*_VV.tif'):
        s1_vv.append(img)

import pickle
with open("s1_vh_list.pkl", 'wb') as f:
    pickle.dump(s1_vh, f)

import pickle
with open("s1_vv_list.pkl", 'wb') as f:
    pickle.dump(s1_vv, f)

###
import pickle
with open("s1_list.pkl", 'wb') as f:
    pickle.dump(s1, f)


import pickle
with open('s1_list.pkl', 'rb') as f:
    s1_list = pickle.load(f)

#'/home/taeil/hpt_k/src/data/'
# 590362

######

######
import argparse


######

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


data_dir = '/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0'

# data_dir = '/home/taeil/SEN12MS/data/bigearthnet/S1'

# data_dir = '/home/cjrd/data/sen12ms_x'

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

# load data
dataset = datasets.ImageFolder(data_dir, transform= data_transforms)
data_loader = DataLoader(dataset=dataset, batch_size=256, num_workers=30, shuffle=True)

def calc_mean_std(loader):
    # var --> std
    channel_sum, channel_sq_sum, num_batch = 0, 0, 0

    for data, _in loader:
        # b x c x h x w
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        channel_sq_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batch += 1

    mean = channel_sum/ num_batch
    std = (channel_sq_sum/ num_batch - mean**2)**0.5

    return mean, std

mean, std = calc_mean_std(data_loader)
print(mean)
print(std)


### new check vh, vv
def calc_mean_std(loader):
    # var --> std
    channel_sum, channel_sq_sum, num_batch = 0, 0, 0
    for data in loader:
        # b x c x h x w
        x1 = data[:, :, :, :]
        channel_sum += torch.mean(x1, dim=[0, 2, 3])
        channel_sq_sum += torch.mean(x1**2, dim=[0, 2, 3])
        num_batch += 1

    mean = channel_sum/ num_batch
    std = (channel_sq_sum/ num_batch - mean**2)**0.5

    return mean, std

mean, std = calc_mean_std(data_loader)
print(mean)
print(std)


###
data_loader = DataLoader(dataset=dataset, batch_size=256, num_workers=30, shuffle=True)
for data, _ in data_loader:
    print(data.shape)


######
######
def main(self):
    print("\n\nbigearth data")

    import torch
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset


    import os
    import pickle as pkl
    from tqdm import tqdm
    import rasterio
    import numpy as np

    path = '/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0/'
    data_index_dir = '/home/taeil/hpt_k/src/data'

    data_transforms = transforms.Compose([
        ToTensor()
    ])

    dataset = bigearthnet(path, data_index_dir, imgTransform=data_transforms,
                      use_s1=True, use_s2=False, use_RGB=True)

    dataset = bigearthnet(path, data_index_dir, imgTransform=False,
                      use_s1=True, use_s2=False, use_RGB=True)

    # s_nor = ds.__getitem__(10)
    # print("id:", s_nor["id"], "\n",
    #       "input shape:", s_nor["image"].shape)

    #

    data_loader = DataLoader(dataset=dataset, batch_size=256, num_workers=30, shuffle=True)
    for data in data_loader:
        print(data.shape)



###

# util function for reading s2 data
def load_s2(path, imgTransform, s2_band):
    bands_selected = s2_band
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    if not imgTransform:
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
    s2 = s2.astype(np.float32)
    return s2

# util function for reading s1 data
def load_s1(path, imgTransform):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    if not imgTransform:
        s1 /= 25
        s1 += 1
    s1 = s1.astype(np.float32)
    return s1

# util function for reading data from single sample
def load_sample(sample, imgTransform, use_s1, use_s2, use_RGB):
    # # load s2 data
    # if use_s2:
    #     img = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_LD)
    # # load only RGB
    # if use_RGB and use_s2 == False:
    #     img = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_RGB)
    #
    # # load s1 data
    # if use_s1:
    #     if use_s2 or use_RGB:
    #         img = np.concatenate((img, load_s1(sample["s1"], imgTransform)), axis=0)
    #     else:
    #         img = load_s1(sample["s1"], imgTransform)

    img = load_s1(sample["s1"], imgTransform)
    # print(sample['id'])
    # print(img)


    # load label
    # lc = labels[sample["id"]]

    # covert label to IGBP simplified scheme
    # if IGBP_s:
    #     cls1 = sum(lc[0:5]);
    #     cls2 = sum(lc[5:7]);
    #     cls3 = sum(lc[7:9]);
    #     cls6 = lc[11] + lc[13];
    #     lc = np.asarray([cls1, cls2, cls3, lc[9], lc[10], cls6, lc[12], lc[14], lc[15], lc[16]])

    # if label_type == "multi_label":
    #     lc_hot = (lc >= threshold).astype(np.float32)
    # else:
    #     loc = np.argmax(lc, axis=-1)
    #     lc_hot = np.zeros_like(lc).astype(np.float32)
    #     lc_hot[loc] = 1

    # rt_sample = {'image': img, 'label': lc_hot, 'id': sample["id"]}

    rt_sample = {'image': img, 'id': sample["id"]}
    # print(rt_sample['image'])
    # print(rt_sample)

    # if imgTransform is not None:
    #     rt_sample = imgTransform(rt_sample)

    return rt_sample['image']


#  calculate number of input channels
def get_ninputs(use_s1, use_s2, use_RGB):
    n_inputs = 0
    if use_s2:
        n_inputs += len(S2_BANDS_LD)
    if use_s1:
        n_inputs += 2
    if use_RGB and use_s2 == False:
        n_inputs += 3

    return n_inputs


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, rt_sample):
        img, sample_id = rt_sample['image'], rt_sample['id']

        rt_sample = {'image': torch.tensor(img), 'id': sample_id}
        return rt_sample


### write a class
class bigearthnet(Dataset):
    """PyTorch dataset class for the bigearth dataset"""

    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val/test split and can be obtained from:
    # https://github.com/MSchmitt1984/SEN12MS/

    def __init__(self, path=None, data_index_dir=None, imgTransform=None, use_s2=False, use_s1=False, use_RGB=False):
        """Initialize the dataset"""

        # inizialize
        super(bigearthnet, self).__init__()
        self.imgTransform = imgTransform

        # make sure input parameters are okay
        if not (use_s2 or use_s1 or use_RGB):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2, s1, RGB] to True!")
        self.use_s2 = use_s2
        self.use_s1 = use_s1
        self.use_RGB = use_RGB

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2, use_RGB)

        # make sure parent dir exists
        assert os.path.exists(path)
        # assert os.path.exists(ls_dir)

        self.samples = []
        # file = os.path.join(data_index_dir, 's1_list.pkl')
        file = os.path.join(data_index_dir, 's1_vh_list.pkl')
        # file = os.path.join(data_index_dir, 's1_vv_list.pkl')
        print(file)
        sample_list = pkl.load(open(file, "rb"))
        # print(sample_list)

        pbar = tqdm(total=len(sample_list))  # 18106 samples in test set
        pbar.set_description("[Load]")

        pbar.set_description("[Load]")

        # for s2_id in sample_list:
        #     mini_name = s2_id.split("_")
        #     s2_loc = os.path.join(path, (mini_name[0] + '_' + mini_name[1]),
        #                           (mini_name[2] + '_' + mini_name[3]), s2_id)
        #     s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
        #
        #     pbar.update()
        #     self.samples.append({"s1": s1_loc, "s2": s2_loc,
        #                          "id": s2_id})

        ######
        for i, name in enumerate(sample_list):
            self.samples.append({"id": i, "s1": name})


        pbar.close()
        # ----------------------------------------------------------------------

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print(f"loaded {len(self.samples)} from {path}")

        # import lables as a dictionary
        # label_file = os.path.join(ls_dir,'IGBP_probability_labels.pkl')

        # a_file = open(label_file, "rb")
        # self.labels = pkl.load(a_file)
        # a_file.close()

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        # labels = self.labels
        return load_sample(sample, self.imgTransform, self.use_s1, self.use_s2, self.use_RGB)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


