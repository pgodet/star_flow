from __future__ import absolute_import, division, print_function

import os
import torch.utils.data as data
from glob import glob

from torchvision import transforms as vision_transforms

from . import transforms
from . import common

import numpy as np
import png

VALIDATE_INDICES_2015 = [10, 11, 12, 25, 26, 30, 31, 40, 41, 42, 46, 52, 53, 72, 73, 74, 75, 76, 80, 81, 85, 86, 95, 96, 97, 98, 104, 116, 117, 120, 121, 126, 127, 153, 172, 175, 183, 184, 190, 199]
VALIDATE_INDICES_2012 = [0, 12, 15, 16, 17, 18, 24, 30, 38, 39, 42, 50, 54, 59, 60, 61, 77, 78, 81, 89, 97, 101, 107, 121, 124, 142, 145, 146, 152, 154, 155, 158, 159, 160, 164, 182, 183, 184, 190]


def read_png_flow(flow_file):
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow[:, :, 0:2], (1 - invalid_idx * 1)[:, :, None]


def kitti_random_crop(im1, im2, flo_f, valid_mask, crop_height=370, crop_width=1224):
    height, width, _ = im1.shape
    # get starting positions
    x = np.random.uniform(0, width - crop_width + 1)
    y = np.random.uniform(0, height - crop_height + 1)
    str_x = int(x)
    str_y = int(y)
    end_x = int(x + crop_width)
    end_y = int(y + crop_height)

    im1 = im1[str_y:end_y, str_x:end_x, :]
    im2 = im2[str_y:end_y, str_x:end_x, :]
    flo_f = flo_f[str_y:end_y, str_x:end_x, :]
    valid_mask = valid_mask[str_y:end_y, str_x:end_x, :]

    return im1, im2, flo_f, valid_mask

def kitti_random_crop_multi(imgs, flo_f, valid_mask, crop_height=370, crop_width=1224):
    height, width, _ = imgs[0].shape
    # get starting positions
    x = np.random.uniform(0, width - crop_width + 1)
    y = np.random.uniform(0, height - crop_height + 1)
    str_x = int(x)
    str_y = int(y)
    end_x = int(x + crop_width)
    end_y = int(y + crop_height)

    for k in range(len(imgs)):
        imgs[k] = imgs[k][str_y:end_y, str_x:end_x, :]
    flo_f = flo_f[str_y:end_y, str_x:end_x, :]
    valid_mask = valid_mask[str_y:end_y, str_x:end_x, :]

    return imgs, flo_f, valid_mask

class KittiMultiframe_comb_test(data.Dataset):
    def __init__(self,
                 args,
                 images_root_2015=None,
                 images_root_2012=None,
                 nframes=5,
                 photometric_augmentations=False,
                 preprocessing_crop=True):

        self._args = args
        self.preprocessing_crop = preprocessing_crop
        self._nframes = nframes

        list_of_indices_2012 = []
        list_of_indices_2015 = []

        # ----------------------------------------------------------
        # KITTI 2015
        # ----------------------------------------------------------        
        if images_root_2015 is not None:

            if not os.path.isdir(images_root_2015):
                raise ValueError("Image directory '%s' not found!")

            all_img_n_2015_filenames = [] # [time][sample]
            for n in range(self._nframes):
                time_index = 12 - self._nframes + n
                all_img_n_2015_filenames.append(sorted(glob(os.path.join(images_root_2015, "*_{:02d}.png".format(time_index)))))

            assert len(all_img_n_2015_filenames[0]) != 0
            for n in range(self._nframes):
                assert len(all_img_n_2015_filenames[0]) == len(all_img_n_2015_filenames[n])
            list_of_indices_2015 = range(len(all_img_n_2015_filenames[0]))           

        # ----------------------------------------------------------
        # KITTI 2012
        # ----------------------------------------------------------        
        if images_root_2012 is not None:

            if not os.path.isdir(images_root_2012):
                raise ValueError("Image directory '%s' not found!")

            all_img_n_2012_filenames = [] # [time][sample]
            for n in range(self._nframes):
                time_index = 12 - self._nframes + n
                all_img_n_2012_filenames.append(sorted(glob(os.path.join(images_root_2012, "*_{:02d}.png".format(time_index)))))

            assert len(all_img_n_2012_filenames[0]) != 0
            for n in range(self._nframes):
                assert len(all_img_n_2012_filenames[0]) == len(all_img_n_2012_filenames[n])
            list_of_indices_2012 = range(len(all_img_n_2012_filenames[0]))

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = []
        self._flow_list = []

        for ii in list_of_indices_2015:

            imgs = []
            for n in range(self._nframes):
                imgs.append(all_img_n_2015_filenames[n][ii])
            idx_0 = os.path.splitext(os.path.basename(imgs[0]))[0][:-3]
            for n in range(self._nframes):
                idx_n = os.path.splitext(os.path.basename(imgs[n]))[0][:-3]
                assert idx_0 == idx_n

            if np.any([not os.path.isfile(im) for im in imgs]):
                continue
            #if not os.path.isfile(im1) or not os.path.isfile(im2):
            #    continue

            self._image_list += [imgs]

        for ii in list_of_indices_2012:

            imgs = []
            for n in range(self._nframes):
                imgs.append(all_img_n_2012_filenames[n][ii])
            idx_0 = os.path.splitext(os.path.basename(imgs[0]))[0][:-3]
            for n in range(self._nframes):
                idx_n = os.path.splitext(os.path.basename(imgs[n]))[0][:-3]
                assert idx_0 == idx_n

            if np.any([not os.path.isfile(im) for im in imgs]):
                continue
            #if not os.path.isfile(im1) or not os.path.isfile(im2):
            #    continue

            self._image_list += [imgs]

        self._size = len(self._image_list)

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

        # read float32 images and flow
        imgs_np0 = [common.read_image_as_byte(filename) for filename in imgs_filenames]

        # possibly apply photometric transformations
        imgs = self._photometric_transform(*imgs_np0)

        # example filename
        #basename = os.path.basename(im1_filename)[:6]
        basenames = [os.path.basename(f)[:9] for f in imgs_filenames]

        example_dict = {
            "input1": imgs[0],
            "input_images": imgs,
            "index": index,
            "basename": basenames,
            "nframes":self._nframes
        }

        return example_dict

    def __len__(self):
        return self._size


class KittiMultiframe_comb(data.Dataset):
    def __init__(self,
                 args,
                 images_root_2015=None,
                 flow_root_2015=None,
                 images_root_2012=None,
                 flow_root_2012=None,
                 nframes=5,
                 photometric_augmentations=False,
                 preprocessing_crop=True,
                 dstype="full"):

        self._args = args
        self.preprocessing_crop = preprocessing_crop
        self._nframes = nframes

        list_of_indices_2012 = []
        list_of_indices_2015 = []

        # ----------------------------------------------------------
        # KITTI 2015
        # ----------------------------------------------------------  
        if images_root_2015 is not None and flow_root_2015 is not None:

            if not os.path.isdir(images_root_2015):
                raise ValueError("Image directory '%s' not found!", images_root_2015)
            if not os.path.isdir(flow_root_2015):
                raise ValueError("Flow directory '%s' not found!", flow_root_2015)

            all_img_n_2015_filenames = [] # [time][sample]
            for n in range(self._nframes):
                time_index = 12 - self._nframes + n
                all_img_n_2015_filenames.append(sorted(glob(os.path.join(images_root_2015, "*_{:02d}.png".format(time_index)))))
            flow_f_2015_filenames = sorted(glob(os.path.join(flow_root_2015, "*_10.png")))

            assert len(all_img_n_2015_filenames[0]) != 0
            for n in range(self._nframes):
                assert len(all_img_n_2015_filenames[0]) == len(all_img_n_2015_filenames[n])
            assert len(flow_f_2015_filenames) == len(all_img_n_2015_filenames[0])

            num_flows_2015 = len(flow_f_2015_filenames)           
            validate_indices_2015 = [x for x in VALIDATE_INDICES_2015 if x in range(num_flows_2015)]            

            if dstype == "train":
                list_of_indices_2015 = [x for x in range(num_flows_2015) if x not in validate_indices_2015]
            elif dstype == "valid":
                list_of_indices_2015 = validate_indices_2015
            elif dstype == "full":
                list_of_indices_2015 = range(len(all_img_n_2015_filenames[0]))
            else:
                raise ValueError("KITTI 2015: dstype '%s' unknown!", dstype)

        # ----------------------------------------------------------
        # KITTI 2012
        # ---------------------------------------------------------- 

        if images_root_2012 is not None and flow_root_2012 is not None:

            if not os.path.isdir(images_root_2012):
                raise ValueError("Image directory '%s' not found!", images_root_2012)
            if not os.path.isdir(flow_root_2012):
                raise ValueError("Flow directory '%s' not found!", flow_root_2012)

            all_img_n_2012_filenames = [] # [time][sample]
            for n in range(self._nframes):
                time_index = 12 - self._nframes + n
                all_img_n_2012_filenames.append(sorted(glob(os.path.join(images_root_2012, "*_{:02d}.png".format(time_index)))))
            flow_f_2012_filenames = sorted(glob(os.path.join(flow_root_2012, "*_10.png")))

            assert len(all_img_n_2012_filenames[0]) != 0
            for n in range(self._nframes):
                assert len(all_img_n_2012_filenames[0]) == len(all_img_n_2012_filenames[n])
            assert len(flow_f_2012_filenames) == len(all_img_n_2012_filenames[0])

            num_flows_2012 = len(flow_f_2012_filenames)           
            validate_indices_2012 = [x for x in VALIDATE_INDICES_2012 if x in range(num_flows_2012)]            

            if dstype == "train":
                list_of_indices_2012 = [x for x in range(num_flows_2012) if x not in validate_indices_2012]
            elif dstype == "valid":
                list_of_indices_2012 = validate_indices_2012
            elif dstype == "full":
                list_of_indices_2012 = range(len(all_img_n_2012_filenames[0]))
            else:
                raise ValueError("KITTI 2012: dstype '%s' unknown!", dstype)


        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = []
        self._flow_list = []

        for ii in list_of_indices_2015:

            imgs = []
            for n in range(self._nframes):
                imgs.append(all_img_n_2015_filenames[n][ii])
            idx_0 = os.path.splitext(os.path.basename(imgs[0]))[0][:-3]
            for n in range(self._nframes):
                idx_n = os.path.splitext(os.path.basename(imgs[n]))[0][:-3]
                assert idx_0 == idx_n

            if np.any([not os.path.isfile(im) for im in imgs]):
                continue
            #if not os.path.isfile(im1) or not os.path.isfile(im2):
            #    continue

            self._image_list += [imgs]

            if dstype is not "test":
                flo_f = flow_f_2015_filenames[ii]
                idx_f = os.path.splitext(os.path.basename(flo_f))[0][:-3]
                assert idx_0 == idx_f                
                if not os.path.isfile(flo_f):
                    continue
                self._flow_list += [[flo_f]]


        for ii in list_of_indices_2012:

            imgs = []
            for n in range(self._nframes):
                imgs.append(all_img_n_2012_filenames[n][ii])
            idx_0 = os.path.splitext(os.path.basename(imgs[0]))[0][:-3]
            for n in range(self._nframes):
                idx_n = os.path.splitext(os.path.basename(imgs[n]))[0][:-3]
                assert idx_0 == idx_n

            if np.any([not os.path.isfile(im) for im in imgs]):
                continue
            #if not os.path.isfile(im1) or not os.path.isfile(im2):
            #    continue

            self._image_list += [imgs]

            if dstype is not "test":
                flo_f = flow_f_2012_filenames[ii]
                idx_f = os.path.splitext(os.path.basename(flo_f))[0][:-3]
                assert idx_0 == idx_f                
                if not os.path.isfile(flo_f):
                    continue
                self._flow_list += [[flo_f]]


        self._size = len(self._image_list)

        assert len(self._image_list) != 0
        if dstype is not "test":
            assert len(self._image_list) == len(self._flow_list)

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
        flo_f_filename = self._flow_list[index][0]

        # read float32 images and flow
        imgs_np0 = [common.read_image_as_byte(filename) for filename in imgs_filenames]
        flo_f_np0, valid_mask = read_png_flow(flo_f_filename)

        if self.preprocessing_crop:
            imgs_np0, flo_f_np0, valid_mask = kitti_random_crop_multi(imgs_np0, flo_f_np0, valid_mask)

        # possibly apply photometric transformations
        imgs = self._photometric_transform(*imgs_np0)

        # convert flow to FloatTensor
        flo_f = common.numpy2torch(flo_f_np0)
        valid_mask_f = common.numpy2torch(valid_mask)

        # example filename
        basenames = [os.path.basename(f)[:9] for f in imgs_filenames]
        #basename = os.path.basename(im1_filename)[:6]

        example_dict = {
            "input1": imgs[0],
            "input_images": imgs,
            "target1": flo_f,
            "index": index,
            "basename": basenames,
            "nframes":self._nframes,
            "input_valid": valid_mask_f
        }

        return example_dict

    def __len__(self):
        return self._size


class KittiMultiframeCombTrain(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=True,
                 preprocessing_crop=True):
        images_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "image_2")
        flow_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "flow_occ")
        images_root_2012 = os.path.join(root, "kitti_flow_2012", "training", "colored_0")
        flow_root_2012 = os.path.join(root, "kitti_flow_2012",  "training", "flow_occ")
        super(KittiMultiframeCombTrain, self).__init__(
            args,
            images_root_2015=images_root_2015,
            flow_root_2015=flow_root_2015,
            images_root_2012=images_root_2012,
            flow_root_2012=flow_root_2012,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="train")


class KittiMultiframeCombVal(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=False,
                 preprocessing_crop=False):
        images_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "image_2")
        flow_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "flow_occ")
        images_root_2012 = os.path.join(root, "kitti_flow_2012", "training", "colored_0")
        flow_root_2012 = os.path.join(root, "kitti_flow_2012",  "training", "flow_occ")
        super(KittiMultiframeCombVal, self).__init__(
            args,
            images_root_2015=images_root_2015,
            flow_root_2015=flow_root_2015,
            images_root_2012=images_root_2012,
            flow_root_2012=flow_root_2012,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="valid")


class KittiMultiframeCombFull(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=True,
                 preprocessing_crop=True):
        images_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "image_2")
        flow_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "flow_occ")
        images_root_2012 = os.path.join(root, "kitti_flow_2012", "training", "colored_0")
        flow_root_2012 = os.path.join(root, "kitti_flow_2012",  "training", "flow_occ")
        super(KittiMultiframeCombFull, self).__init__(
            args,
            images_root_2015=images_root_2015,
            flow_root_2015=flow_root_2015,
            images_root_2012=images_root_2012,
            flow_root_2012=flow_root_2012,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="full")


class KittiMultiframeComb2015Train(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=True,
                 preprocessing_crop=True):
        images_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "image_2")
        flow_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "flow_occ")
        super(KittiMultiframeComb2015Train, self).__init__(
            args,
            images_root_2015=images_root_2015,
            flow_root_2015=flow_root_2015,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="train")


class KittiMultiframeComb2015Val(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=False,
                 preprocessing_crop=False):
        images_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "image_2")
        flow_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "flow_occ")
        super(KittiMultiframeComb2015Val, self).__init__(
            args,
            images_root_2015=images_root_2015,
            flow_root_2015=flow_root_2015,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="valid")


class KittiMultiframeComb2015Full(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=True,
                 preprocessing_crop=True):
        images_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "image_2")
        flow_root_2015 = os.path.join(root, "kitti_flow_2015", "training", "flow_occ")
        super(KittiMultiframeComb2015Full, self).__init__(
            args,
            images_root_2015=images_root_2015,
            flow_root_2015=flow_root_2015,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="full")


class KittiMultiframeComb2015Test(KittiMultiframe_comb_test):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=False,
                 preprocessing_crop=False):
        images_root_2015 = os.path.join(root, "kitti_flow_2015", "testing", "image_2")
        super(KittiMultiframeComb2015Test, self).__init__(
            args,
            images_root_2015=images_root_2015,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop)


class KittiMultiframeComb2012Train(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=True,
                 preprocessing_crop=True):
        images_root_2012 = os.path.join(root, "kitti_flow_2012", "training", "colored_0")
        flow_root_2012 = os.path.join(root, "kitti_flow_2012",  "training", "flow_occ")
        super(KittiMultiframeComb2012Train, self).__init__(
            args,
            images_root_2012=images_root_2012,
            flow_root_2012=flow_root_2012,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="train")


class KittiMultiframeComb2012Val(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=False,
                 preprocessing_crop=False):
        images_root_2012 = os.path.join(root, "kitti_flow_2012", "training", "colored_0")
        flow_root_2012 = os.path.join(root, "kitti_flow_2012",  "training", "flow_occ")
        super(KittiMultiframeComb2012Val, self).__init__(
            args,
            images_root_2012=images_root_2012,
            flow_root_2012=flow_root_2012,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="valid")


class KittiMultiframeComb2012Full(KittiMultiframe_comb):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=True,
                 preprocessing_crop=True):
        images_root_2012 = os.path.join(root, "kitti_flow_2012", "training", "colored_0")
        flow_root_2012 = os.path.join(root, "kitti_flow_2012",  "training", "flow_occ")
        super(KittiMultiframeComb2012Full, self).__init__(
            args,
            images_root_2012=images_root_2012,
            flow_root_2012=flow_root_2012,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop,
            dstype="full")


class KittiMultiframeComb2012Test(KittiMultiframe_comb_test):
    def __init__(self,
                 args,
                 root,
                 nframes=5,
                 photometric_augmentations=False,
                 preprocessing_crop=False):
        images_root_2012 = os.path.join(root, "kitti_flow_2012", "testing", "colored_0")
        super(KittiMultiframeComb2012Test, self).__init__(
            args,
            images_root_2012=images_root_2012,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            preprocessing_crop=preprocessing_crop)