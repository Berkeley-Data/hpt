
"""
### Tsung-Chin Han
### Descritions -
This script is to help calculate for "each" of the band mean and standard deviation for
BigEarthNet dataset. Note that this custom load does not stack all channels together.
Instead, by specifying the channel name, it helps compute the band mean and standard deviation for each channel data.


### BigEarthNet channels for S-1 and S-2
# S-1 - VV, VH
# S-2 - B01, B02, B03, B04, B05, B06, B07, B08, B09, B11, B12, B8A

Note that the calc is based on raw pixel values to reflect the original data stats,
It does not do any pixel "clip" or normalization.

"""

import argparse
import os
import pickle as pkl
from tqdm import tqdm
import rasterio
import numpy as np
import glob

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser(description='Compute band image mean and stdev from BigEarthNet class')
# batch
parser.add_argument('--numworkers', type=int, default=30)
parser.add_argument('--batchsize', type=int, default=1)

# data path & data index directory
parser.add_argument('--path', type=str, default=None,
                    help='path to BigEarthNet dataset')
parser.add_argument('--data_index_dir', type=str, default=None,
                    help="path to generated data list")
# data modality
parser.add_argument('--use_s1', action='store_true', default=False,
                    help='use Sentinel-1 data')
parser.add_argument('--use_s2', action='store_true', default=False,
                    help='use Sentinel-2 bands')
# channel name
parser.add_argument('--band', type=str, default=None,
                    help='band(channel) name from BigEarthNet dataset')

# band (channel) name
# S1 - VV, VH
# S2 - B01, B02, B03, B04, B05, B06, B07, B08, B09, B11, B12, B8A
# band info
s1_band = ['VV', 'VW']
s2_band = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']


def calc_mean_std(loader):
    """calc band image dataset mean and standard deviation
    """
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


def load_s1(path, imgTransform):
    """load s1 band data raw
    """
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    return s1


def load_s2(path, imgTransform):
    """load s2 band data raw
    """
    with rasterio.open(path) as data:
        s2 = data.read()
    s2 = s2.astype(np.float32)
    s2 = np.nan_to_num(s2)
    return s2


def load_sample(sample, imgTransform, use_s1, use_s2):
    """util to load sample data (wip)
        to do --> stacked data
    """
    if use_s1:
        img = load_s1(sample["s1"], imgTransform)
    if use_s2:
        img = load_s2(sample["s2"], imgTransform)

    rt_sample = {'image': img, 'id': sample["id"]}

    return rt_sample['image']


def get_ninputs(use_s1, use_s2):
    """return number of input channels
        wip - to do - work on stacked the bands
    """
    n_inputs = 0
    if use_s1:
        n_inputs += len(s1_band)
    if use_s2:
        n_inputs += len(s2_band)

    return n_inputs


class ToTensor(object):
    """convert sample ndarrays to inpiut Tensors."""

    def __call__(self, rt_sample):
        img, sample_id = rt_sample['image'], rt_sample['id']

        rt_sample = {'image': torch.tensor(img), 'id': sample_id}

        return rt_sample


### custom BigEarthNet dataset class
class bigearthnet(Dataset):
    """pytorch dataset class custom for BigEarthNet
    """

    def __init__(self, path=None, data_index_dir=None, imgTransform=None, use_s2=False, use_s1=False, band=None):
        """Initialize the dataset
        """

        # initialize
        super(bigearthnet, self).__init__()
        self.imgTransform = imgTransform

        # make sure input parameters are okay
        if not (use_s2 or use_s1):
            raise ValueError("input error, please check to data modality")
        self.use_s1 = use_s1
        self.use_s2 = use_s2

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2)

        # get sample images
        self.samples = []
        if use_s1 and band in s1_band:
            file = os.path.join(data_index_dir, 's1_'+band+'_list.pkl')

        if use_s2 and band in s2_band:
            file = os.path.join(data_index_dir, 's2_'+band+'_list.pkl')

        sample_list = pkl.load(open(file, "rb"))
        pbar = tqdm(total=len(sample_list))
        pbar.set_description("[Loading Images]")
        for i, name in enumerate(sample_list):
            if use_s1:
                self.samples.append({"id": i, "s1": name})
            if use_s2:
                self.samples.append({"id": i, "s2": name})
            pbar.update()

        pbar.close()

        return


    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        # labels = self.labels
        return load_sample(sample, self.imgTransform, self.use_s1, self.use_s2)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


def file_list_generator(path, band_list, use_s1=False, use_s2=False):
    """"generate band file list from s-1 or s-2
    """
    # band info
    # s1_band = ['VV', 'VW']
    # s2_band = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']

    import glob
    import os
    import pickle

    for band in band_list:
        tmp = []
        # s1 - "/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0
        # s2 - "/home/cjrd/data/bigearthnet/BigEarthNet-v1.0
        for file in glob.glob(path+'/*'):
            for img in glob.glob(file+'/*_'+band+'.tif'):
                print(img)
                tmp.append(img)
        if use_s1:
            with open("s1_"+band+"_list.pkl", 'wb') as f:
                pickle.dump(tmp, f)
        if use_s2:
            with open("s2_"+band+"_list.pkl", 'wb') as f:
                pickle.dump(tmp, f)
        del tmp

    return

def main(args):
    # path = '/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0/'
    # data_index_dir = '/home/taeil/hpt_k/src/data'

    # band info
    # s1_band = ['VV', 'VW']
    # s2_band = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']

    data_transforms = transforms.Compose([
        ToTensor()
    ])

    dataset = bigearthnet(path=args.path, data_index_dir=args.data_index_dir, imgTransform=data_transforms,\
                          use_s1=args.use_s1, use_s2=args.use_s2, band=args.band)

    data_loader = DataLoader(dataset=dataset, batch_size=args.batchsize, num_workers=args.numworkers, shuffle=True)

    # calc mean and std for the dataset
    mean, std = calc_mean_std(data_loader)
    print('band:{} -- mean:{}, std:{}'.format(args.band, mean[0], std[0]))


    ####### to-do later--->
    # 1. stacked the bands dataset -- if needed to load all at once


if __name__ == "__main__":
    main(parser.parse_args())

