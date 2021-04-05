
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_dir = '/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0'
# data_dir = '/home/cjrd/data/sen12ms_x'


# load data
dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
data_loader = DataLoader(dataset=dataset, batch_size=512, num_workers=30, shuffle=True)

def calc_mean_std(loader):
    # var --> std
    channel_sum, channel_sq_sum, num_batch = 0, 0, 0

    for data, _ in loader:
        # b x c x w x h
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        channel_sq_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batch += 1

    mean = channel_sum/ num_batch
    std = (channel_sq_sum/ num_batch - mean**2)**0.5

    return mean, std

mean, std = calc_mean_std(data_loader)
print(mean)
print(std)


### back calc band mean and std
# def back_calc(loader):
#
#     channel_sum, channel_sq_sum, num_batch = 0, 0, 0
#     for data, _ in loader:
#         x1 = data[0, 0]
#         print('x1', x1.shape)
#         channel_sum += torch.mean(x1, dim=[0, 1])
#         channel_sq_sum += torch.mean(x1 ** 2, dim=[0, 1])
#         num_batch += 1
#
#     mean = channel_sum / num_batch
#     std = (channel_sq_sum / num_batch - mean ** 2) ** 0.5
#
#     return mean, std
#
# mean, std = calc_mean_std(data_loader)
# print(mean)
# print(std)
#







