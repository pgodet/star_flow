from __future__ import absolute_import, division, print_function

import os
import torch
import torch.utils.data as data
from glob import glob

import torch
from torchvision import transforms as vision_transforms

from . import transforms
from . import common

import numpy as np


def fillingInNaN(flow):
    h, w, c = flow.shape
    indices = np.argwhere(np.isnan(flow))
    neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for ii, idx in enumerate(indices):
        sum_sample = 0
        count = 0
        for jj in range(0, len(neighbors) - 1):
            hh = idx[0] + neighbors[jj][0]
            ww = idx[1] + neighbors[jj][1]
            if hh < 0 or hh >= h:
                continue
            if ww < 0 or ww >= w:
                continue
            sample_flow = flow[hh, ww, idx[2]]
            if np.isnan(sample_flow):
                continue
            sum_sample += sample_flow
            count += 1
        if count is 0:
            print('FATAL ERROR: no sample')
        flow[idx[0], idx[1], idx[2]] = sum_sample / count

    return flow


class FlyingThings3dMultiframe(data.Dataset):
    def __init__(self,
                 args,
                 images_root,
                 flow_root,
                 occ_root,
                 seq_lengths_path, nframes=5,
                 photometric_augmentations=False,
                 backward=False):

        self._args = args
        self._nframes = nframes
        self.backward = backward

        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!", images_root)
        if flow_root is not None and not os.path.isdir(flow_root):
            raise ValueError("Flow directory '%s' not found!", flow_root)
        if occ_root is not None and not os.path.isdir(occ_root):
            raise ValueError("Occ directory '%s' not found!", occ_root)

        if flow_root is not None:
            flow_f_filenames = sorted(glob(os.path.join(flow_root, "into_future/*.flo")))
            flow_b_filenames = sorted(glob(os.path.join(flow_root, "into_past/*.flo")))

        if occ_root is not None:
            occ1_filenames = sorted(glob(os.path.join(occ_root, "into_future/*.png")))
            occ2_filenames = sorted(glob(os.path.join(occ_root, "into_past/*.png")))

        all_img_filenames = sorted(glob(os.path.join(images_root, "*.png")))

        self._image_list = []
        self._flow_list = [] if flow_root is not None else None
        self._occ_list = [] if occ_root is not None else None

        assert len(all_img_filenames) != 0
        assert len(flow_f_filenames) != 0
        assert len(flow_b_filenames) != 0
        assert len(occ1_filenames) != 0
        assert len(occ2_filenames) != 0

        self._seq_lengths = np.load(seq_lengths_path)

        ## path definition
        path_flow_f = os.path.join(flow_root, "into_future")
        path_flow_b = os.path.join(flow_root, "into_past")
        path_occ_f = os.path.join(occ_root, "into_future")
        path_occ_b = os.path.join(occ_root, "into_past")

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------

        idx_first = 0

        for seq_len in self._seq_lengths:
            list_images = []
            list_flows = []
            list_occs = []

            for ii in range(idx_first, idx_first + seq_len - 1):
                list_images.append(os.path.join(images_root, "{:07d}".format(ii) + ".png"))
                if self.backward:
                    list_flows.append(os.path.join(path_flow_b, "{:07d}".format(ii+1) + ".flo"))
                    list_occs.append(os.path.join(path_occ_b, "{:07d}".format(ii+1) + ".png"))
                else:
                    list_flows.append(os.path.join(path_flow_f, "{:07d}".format(ii) + ".flo"))
                    list_occs.append(os.path.join(path_occ_f, "{:07d}".format(ii) + ".png"))
                    #if not os.path.isfile(flo_f) or not os.path.isfile(flo_b) or not os.path.isfile(im1) or not os.path.isfile(
                    #        im2) or not os.path.isfile(occ1) or not os.path.isfile(occ2):
                    #    continue
            list_images.append(os.path.join(images_root, "{:07d}".format(ii + 1) + ".png")) # ii + 1 = idx_first + seq_len - 1

            for i in range(len(list_images) - self._nframes + 1):

                imgs = list_images[i:i+self._nframes]
                flows = list_flows[i:i+self._nframes-1]
                occs = list_occs[i:i+self._nframes-1]

                self._image_list += [imgs]
                self._flow_list += [flows]
                self._occ_list += [occs]

            idx_first += seq_len

        self._size = len(self._image_list)

        assert len(self._image_list) == len(self._flow_list)
        assert len(self._occ_list) == len(self._flow_list)
        assert len(self._image_list) != 0


        # ----------------------------------------------------------
        # photometric_augmentations
        # ----------------------------------------------------------
        if photometric_augmentations:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> PIL
                vision_transforms.ToPILImage(),
                # PIL -> PIL : random hsv and contrast
                vision_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # PIL -> FloatTensor
                vision_transforms.transforms.ToTensor(),
                transforms.RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True),
            ], from_numpy=True, to_numpy=False)

        else:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> FloatTensor
                vision_transforms.transforms.ToTensor(),
            ], from_numpy=True, to_numpy=False)

    def __getitem__(self, index):

        index = index % self._size

        imgs_filenames = self._image_list[index]
        flows_filenames = self._flow_list[index]
        occs_filenames = self._occ_list[index]

        # read float32 images and flow
        imgs_np0 = [common.read_image_as_byte(filename) for filename in imgs_filenames]
        flows_np0 = [common.read_flo_as_float32(filename) for filename in flows_filenames]
        occs_np0 = [common.read_occ_image_as_float32(filename) for filename in occs_filenames]

        # temp - check isnan
        for ii in range(len(flows_np0)):
            if np.any(np.isnan(flows_np0[ii])):
                flows_np0[ii] = fillingInNaN(flows_np0[ii])

        # possibly apply photometric transformations
        imgs = self._photometric_transform(*imgs_np0)

        # convert flow to FloatTensor
        flows = [common.numpy2torch(flo_np0) for flo_np0 in flows_np0]

        # convert occ to FloatTensor
        occs = [common.numpy2torch(occ_np0) for occ_np0 in occs_np0]

        # example filename
        basename = [os.path.basename(filename)[:-4] for filename in imgs_filenames]

        example_dict = {
            "input1": imgs[0],
            "input_images": imgs, # "target_flows": torch.stack(flows, 0),
            "target1": flows[0],
            "target_flows": flows, #torch.stack(flows, 0)
            "target_occ1": occs[0],
            "target_occs": occs, #torch.stack(occs, 0)
            "index": index,
            "basename": basename,
            "nframes": self._nframes
        }

        return example_dict

    def __len__(self):
        return self._size


# class FlyingThings3dFinalTrain(FlyingThings3d):
#     def __init__(self,
#                  args,
#                  root,
#                  photometric_augmentations=True):
#         images_root = os.path.join(root, "frames_finalpass")
#         flow_root = os.path.join(root, "optical_flow")
#         occ_root = os.path.join(root, "occlusion")
#         seq_lengths_path = os.path.join(root, "seq_lengths.npy")
#         super(FlyingThings3dFinalTrain, self).__init__(
#             args,
#             images_root=images_root,
#             flow_root=flow_root,
#             occ_root=occ_root,
#             seq_lengths_path=seq_lengths_path,
#             photometric_augmentations=photometric_augmentations)


# class FlyingThings3dFinalTest(FlyingThings3d):
#     def __init__(self,
#                  args,
#                  root,
#                  photometric_augmentations=False):
#         images_root = os.path.join(root, "frames_finalpass")
#         flow_root = os.path.join(root, "optical_flow")
#         occ_root = os.path.join(root, "occlusion")
#         seq_lengths_path = os.path.join(root, "seq_lengths.npy")
#         super(FlyingThings3dFinalTest, self).__init__(
#             args,
#             images_root=images_root,
#             flow_root=flow_root,
#             occ_root=occ_root,
#             seq_lengths_path=seq_lengths_path,
#             photometric_augmentations=photometric_augmentations)


class FlyingThings3dMultiframeCleanTrain(FlyingThings3dMultiframe):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=True,
                 backward=False):
        images_root = os.path.join(root, "train", "image_clean", "left")
        flow_root = os.path.join(root, "train", "flow", "left")
        occ_root = os.path.join(root, "train", "flow_occlusions", "left")
        seq_lengths_path = os.path.join(root, "train", "seq_lengths.npy")
        super(FlyingThings3dMultiframeCleanTrain, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            occ_root=occ_root,
            seq_lengths_path=seq_lengths_path,
            photometric_augmentations=photometric_augmentations,
            nframes=nframes, backward=backward)


class FlyingThings3dMultiframeCleanTest(FlyingThings3dMultiframe):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=False,
                 backward=False):
        images_root = os.path.join(root, "val", "image_clean", "left")
        flow_root = os.path.join(root, "val", "flow", "left")
        occ_root = os.path.join(root, "val", "flow_occlusions", "left")
        seq_lengths_path = os.path.join(root, "val", "seq_lengths.npy")
        super(FlyingThings3dMultiframeCleanTest, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            occ_root=occ_root,
            seq_lengths_path=seq_lengths_path,
            photometric_augmentations=photometric_augmentations,
            nframes=nframes, backward=backward)
