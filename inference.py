import os
import sys
from glob import glob

import scipy.misc
import numpy as np
import torch

from torchvision import transforms as vision_transforms
import models
from datasets import common
from configuration import ModelAndLoss

from utils.flow import flow_to_png_middlebury, write_flow

import pylab as pl
pl.interactive(True)

import argparse

'''
Example (will save results in ./output/):
python inference.py \
  --model StarFlow \
  --checkpoint saved_checkpoint/StarFlow_things/checkpoint_best.ckpt \
  --data-root /data/mpisintelcomplete/training/final/ambush_6/ \
  --file-list frame_0004.png frame_0005.png frame_0006.png frame_0007.png
'''

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", type=str, default="./output")
parser.add_argument("--data-root", type=str,
                    default="./")
parser.add_argument('--file-list', nargs='*', default=[-1], type=str)

parser.add_argument("--model", type=str, default='StarFlow')
parser.add_argument('--checkpoint', dest='checkpoint', default=None,
                    metavar='PATH', help='path to pre-trained model')

parser.add_argument('--device', type=int, default=0)
parser.add_argument("--no-cuda", action="store_true")

args = parser.parse_args()

# use cuda GPU
use_cuda = (not args.no_cuda) and torch.cuda.is_available()

# ---------------------
# Load pretrained model
# ---------------------
MODEL = models.__dict__[args.model]
net = ModelAndLoss(None, MODEL(None), None)
checkpoint_with_state = torch.load(args.checkpoint,
                                map_location=lambda storage,
                                loc: storage.cuda(args.device))
state_dict = checkpoint_with_state['state_dict']
net.load_state_dict(state_dict)
net.eval()
net.cuda()

# -------------------
# Load image sequence
# -------------------
if not os.path.exists(args.data_root):
    raise ValueError("data-root: {} not found".format(args.data_root))
if len(args.file_list) == 0:
    raise ValueError("file-list empty")
elif len(args.file_list) == 1:
    path = os.path.join(args.data_root, args.file_list[0])
    list_path_imgs = sorted(glob(path))
    if len(list_path_imgs) == 0:
        raise ValueError("no data were found")
else:
    list_path_imgs = [os.path.join(args.data_root, file_name)
                                for file_name in args.file_list]
    for path_im in list_path_imgs:
        if not os.path.isfile(path_im):
            raise ValueError("file {} not found".format(path_im))
img_reader = common.read_image_as_byte
#flo_reader = common.read_flo_as_float32
imgs_np = [img_reader(path) for path in list_path_imgs]
if imgs_np[0].squeeze().ndim == 2:
    imgs_np = [np.dstack([im]*3) for im in imgs_np]
to_tensor = vision_transforms.ToTensor()
images = [to_tensor(im).unsqueeze(0).cuda() for im in imgs_np]
input_dict = {'input_images':images}

# ---------------
# Flow estimation
# ---------------
with torch.no_grad():
    output_dict = net._model(input_dict)

estimated_flow = output_dict['flow']

if len(imgs_np) > 2:
    estimated_flow_np = estimated_flow[:,0].cpu().numpy()
    estimated_flow_np = [flow for flow in estimated_flow_np]
else:
    estimated_flow_np = [estimated_flow[0].cpu().numpy()]


# ------------
# Save results
# ------------
if not os.path.exists(os.path.join(args.savedir, "visu")):
    os.makedirs(os.path.join(args.savedir, "visu"))
if not os.path.exists(os.path.join(args.savedir, "flow")):
    os.makedirs(os.path.join(args.savedir, "flow"))
for t in range(len(imgs_np)-1):
    flow_visu = flow_to_png_middlebury(estimated_flow_np[t])
    basename = os.path.splitext(os.path.basename(list_path_imgs[t]))[0]
    file_name_flow_visu = os.path.join(args.savedir, 'visu',
                                basename + '_flow_visu.png')
    file_name_flow = os.path.join(args.savedir, 'flow',
                                basename + '_flow.flo')
    scipy.misc.imsave(file_name_flow_visu, flow_visu)
    write_flow(file_name_flow, estimated_flow_np[t].swapaxes(0, 1).swapaxes(1, 2))
