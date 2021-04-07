
"""
- Tsung-Chin Han (Ken)-

- Descritions -
This script is to help calculate for each of the band mean and standard deviation for
BigEarthNet dataset. Note that this custom load does not stack all channels together.
Instead, by specifying the channel name, it helps compute the band mean and standard deviation.
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
parser.add_argument('--numworkers', type=int, default=30)
parser.add_argument('--batchsize', type=int, default=1)

# data path & data index directory
parser.add_argument('--path', type=str, default=None,
                    help='path to BigEarthNet dataset')
parser.add_argument('--data_index_dir', type=str, default=None,
                    help="path to generated data list")
# data modality
parser.add_argument('--use_s1', action='store_true', default=True,
                    help='use sentinel-1 data')
parser.add_argument('--use_s2', action='store_true', default=False,
                    help='use sentinel-2 bands')
# band (channel) name
# S1 - VV, VH
# S2 - B01, B02, B03, B04, B05, B06, B07, B08, B09, B11, B12, B8A
parser.add_argument('--band', type=str, default=None,
                    help='band(channel) name from BigEarthNet dataset')

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
    """util to load s1 data
    """
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


def load_s2(path, imgTransform, s2_band):
    """wip
    """
    # bands_selected = s2_band
    # with rasterio.open(path) as data:
    #     s2 = data.read(bands_selected)
    # s2 = s2.astype(np.float32)
    # if not imgTransform:
    #     s2 = np.clip(s2, 0, 10000)
    #     s2 /= 10000
    # s2 = s2.astype(np.float32)
    # return s2


def load_sample(sample, imgTransform, use_s1, use_s2):
    """util to load sample data (wip)
    """
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

    if use_s1:
        img = load_s1(sample["s1"], imgTransform)


    # rt_sample = {'image': img, 'label': lc_hot, 'id': sample["id"]}

    rt_sample = {'image': img, 'id': sample["id"]}


    return rt_sample['image']


#  calculate number of input channels
def get_ninputs(use_s1, use_s2):
    n_inputs = 0
    # if use_s2:
    #     n_inputs += len(S2_BANDS_LD)
    if use_s1:
        n_inputs += 2

    return n_inputs


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

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
            raise ValueError("No input specified, set at least one of " + "use_[s2, s1, RGB] to True!")
        self.use_s2 = use_s2
        self.use_s1 = use_s1

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2)

        self.samples = []
        # file = os.path.join(data_index_dir, 's1_list.pkl')

        if band == 'VV':
            file = os.path.join(data_index_dir, 's1_vv_list.pkl')
        elif band == 'VH':
            file = os.path.join(data_index_dir, 's1_vh_list.pkl')

        sample_list = pkl.load(open(file, "rb"))

        pbar = tqdm(total=len(sample_list))
        pbar.set_description("[Load]")

        for i, name in enumerate(sample_list):
            self.samples.append({"id": i, "s1": name})
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



def main(args):
    # path = '/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0/'
    # data_index_dir = '/home/taeil/hpt_k/src/data'

    data_transforms = transforms.Compose([
        ToTensor()
    ])

    dataset = bigearthnet(path=args.path, data_index_dir=args.data_index_dir, imgTransform=data_transforms,\
                          use_s1=args.use_s1, use_s2=args.use_s2, band=args.band)

    data_loader = DataLoader(dataset=dataset, batch_size=args.batchsize, num_workers=args.numworkers, shuffle=True)

    # calc mean and std for the dataset
    mean, std = calc_mean_std(data_loader)
    print('mean:{}, std:{}'.format(mean[0], std[0]))



if __name__ == "__main__":
    main(parser.parse_args())

