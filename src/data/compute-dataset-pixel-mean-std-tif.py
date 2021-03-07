
#!/usr/bin/env python
"""
This utility computes the mean, var, and std of pixel values of a large image dataset
e.g.
./compute-dataset-pixel-mean-std.py --data /path/to/image-folder

where image-folder has the structure from ImageFolder in pytorch
class/image-name.jp[e]g
or whatever image extension you're using
"""

import sys
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as T

from dataset import SEN12MS, ToTensor, Normalize

parser = argparse.ArgumentParser(description='Compute image statistics from ImageFolder')
# parser.add_argument('--data', metavar='DIR', help='path to image directory with structure class/image.ext', required=True)
parser.add_argument('--numworkers', type=int, default=30)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--numbatches', type=int, default=-1)


# # data directory
parser.add_argument('--data_dir', type=str, default=None,
                    help='path to SEN12MS dataset')
parser.add_argument('--data_index_dir', type=str, default=None,
                    help="path to label data and split list")

# input/output
parser.add_argument('--use_s2', action='store_true', default=False,
                    help='use sentinel-2 bands')
parser.add_argument('--use_s1', action='store_true', default=False,
                    help='use sentinel-1 data')
parser.add_argument('--use_RGB', action='store_true', default=True,
                    help='use sentinel-2 RGB bands')
# parser.add_argument('--IGBP_simple', action='store_true', default=True,
#                     help='use IGBP simplified scheme; otherwise: IGBP original scheme')
# parser.add_argument('--label_type', type=str, choices = label_choices,
#                     default='multi_label',
#                     help="label-type (default: multi_label)")
# parser.add_argument('--threshold', type=float, default=0.1,
#                     help='threshold to convert probability-labels to multi-hot \
#                     labels, mean/std for normalizatin would not be accurate \
#                     if the threshold is larger than 0.22. \
#                     for single_label threshold would be ignored')


def main(args):
    # dataset = datasets.ImageFolder(args.data ,
    #                                transform=transforms.Compose([transforms.ToTensor(),
    #                                                              transforms.Lambda(lambda x:  torch.stack([x.mean([1,2]), (x*x).mean([1,2])]) )])) # x.view(x.shape[0], -1))]))

    # bands_mean = {'s1_mean': [-11.76858, -18.294598],
    #               's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
    #                           2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}
    #
    # bands_std = {'s1_std': [4.525339, 4.3586307],
    #              's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
    #                         1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]}

    # , T.Normalize(bands_mean, bands_std)

    # load datasets
    # imgTransform = T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.stack([x.mean([1,2]), (x*x).mean([1,2])]) )])
    imgTransform = T.Compose([ToTensor()])

    dataset = SEN12MS(args.data_dir,
                      args.data_index_dir,
                      imgTransform=imgTransform,
                      # label_type=label_type,
                      # threshold=args.threshold,
                      # subset="small",
                      use_s1=args.use_s1,
                      use_s2=args.use_s2,
                      use_RGB=args.use_RGB)

    loader = DataLoader(
        dataset,
        # batch_size=args.batchsize,
        num_workers=args.numworkers,
        shuffle=True
    )

    mean = 0.
    nb_samples = 0.
    results = torch.zeros((2,1))   # torch.Size([1, 3, 256, 256])
    N = len(dataset)
    Nproc = 0
    i = 0
    if args.numbatches < 0:
        NB = len(loader)
    else:
        NB = args.numbatches

    for sample in loader:
        x = sample["image"]
        data = torch.stack([x.mean([1,2]), (x*x).mean([1,2])])

        results += data.sum(2)
        Nproc += data.shape[0]
        i += 1
        print("batch: {}/{}".format(i, NB))
        if i >= NB:
            break

    print(results)
    means = results[0,:] / Nproc
    sqsums = results[1,:] / Nproc
    vvars = sqsums - means**2
    print('means: {}'.format(means))
    print('vars: {}'.format(vvars))
    stds = vvars**0.5
    print('stds: {}'.format(stds))


if __name__ == "__main__":
    main(parser.parse_args())
