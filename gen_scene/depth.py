import torch
from gen_scene.data import NYUv2
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

t = transforms.Compose([])
dataset = NYUv2(root="data/NYUv2", download=True, rgb_transform=t,
                seg_transform=t, sn_transform=t, depth_transform=t)

rgb1, seg1, normal1, depth1 = dataset[140]
rgb2, seg2, normal2, depth2 = dataset[141]

rgb1.save('test1.png')
rgb2.save('test2.png')

mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
rgb = np.array(rgb1)
rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
rgb = torch.div((rgb - mean), std)
rgb = rgb[None, :, :, :].cuda()

model = torch.hub.load(
    'yvanyin/metric3d',
    'metric3d_vit_small',
    pretrain=True
)
model = model.cuda()

pred_depth, confidence, output_dict = model.inference(
    {'input': rgb}
)

pred_depth
