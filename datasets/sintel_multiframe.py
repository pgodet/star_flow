from __future__ import absolute_import, division, print_function

import os
import torch.utils.data as data
from glob import glob

from torchvision import transforms as vision_transforms

from . import transforms
from . import common

import tools


VALIDATE_INDICES = [
    199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 217, 340, 341, 342, 343, 344,
    345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
    357, 358, 359, 360, 361, 362, 363, 364, 536, 537, 538, 539,
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551,
    552, 553, 554, 555, 556, 557, 558, 559, 560, 659, 660, 661,
    662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673,
    674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
    686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,
    967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978,
    979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990,
    991]


class _SintelMultiframe(data.Dataset):
    def __init__(self,
                 args,
                 dir_root=None, nframes=5,
                 photometric_augmentations=False,
                 imgtype=None,
                 dstype=None, reverse=False):

        self._args = args
        self._nframes = nframes
        self.reverse = reverse

        images_root = os.path.join(dir_root, imgtype)
        if imgtype is "comb":
            images_root = os.path.join(dir_root, "clean")
        flow_root = os.path.join(dir_root, "flow")
        occ_root = os.path.join(dir_root, "occlusions_rev")

        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!", images_root)
        if flow_root is not None and not os.path.isdir(flow_root):
            raise ValueError("Flow directory '%s' not found!", flow_root)
        if occ_root is not None and not os.path.isdir(occ_root):
            raise ValueError("Occ directory '%s' not found!", occ_root)

        all_flo_filenames = sorted(glob(os.path.join(flow_root, "*/*.flo")))
        all_occ_filenames = sorted(glob(os.path.join(occ_root, "*/*.png")))
        all_img_filenames = sorted(glob(os.path.join(images_root, "*/*.png")))

        # Remember base for substraction at runtime
        # e.g. subtract_base = "/home/user/.../MPI-Sintel-Complete/training/clean"
        self._substract_base = tools.cd_dotdot(images_root)

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = tools.cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))

        self._image_list = []
        self._flow_list = []
        self._occ_list = []

        for base_folder in base_folders:
            img_filenames = [x for x in all_img_filenames if base_folder in x]
            flo_filenames = [x for x in all_flo_filenames if base_folder in x]
            occ_filenames = [x for x in all_occ_filenames if base_folder in x]

            if self._nframes >= 0:
                for i in range(len(img_filenames) - self._nframes + 1):
                    imgs = img_filenames[i:i+self._nframes]
                    flows = flo_filenames[i:i+self._nframes-1]
                    occs = occ_filenames[i:i+self._nframes-1]
                    if self.reverse:
                        imgs = imgs[::-1]
                        flows = flows[::-1]
                        occs = occs[::-1]
                    self._image_list += [imgs]
                    self._flow_list += [flows]
                    self._occ_list += [occs]

                    # Sanity check
                    for k in range(len(flows)):
                        im1_base_filename = os.path.splitext(os.path.basename(imgs[k]))[0]
                        im2_base_filename = os.path.splitext(os.path.basename(imgs[k+1]))[0]
                        flo_base_filename = os.path.splitext(os.path.basename(flows[k]))[0]
                        occ_base_filename = os.path.splitext(os.path.basename(occs[k]))[0]
                        im1_frame, im1_no = im1_base_filename.split("_")
                        im2_frame, im2_no = im2_base_filename.split("_")
                        assert(im1_frame == im2_frame)
                        if self.reverse:
                            assert(int(im2_no) == int(im1_no) - 1)
                        else:
                            assert(int(im1_no) == int(im2_no) - 1)

                        flo_frame, flo_no = flo_base_filename.split("_")
                        assert(im1_frame == flo_frame)
                        if self.reverse:
                            assert(int(im2_no) == int(flo_no))
                        else:
                            assert(int(im1_no) == int(flo_no))

                        occ_frame, occ_no = occ_base_filename.split("_")
                        assert(im1_frame == occ_frame)
                        if self.reverse:
                            assert(int(im2_no) == int(occ_no))
                        else:
                            assert(int(im1_no) == int(occ_no))

            else:
                imgs = img_filenames
                flows = flo_filenames
                occs = occ_filenames
                self._image_list += [imgs]
                self._flow_list += [flows]
                self._occ_list += [occs]

        assert len(self._image_list) == len(self._flow_list)
        assert len(self._image_list) == len(self._occ_list)

        # -------------------------------------------------------------
        # Remove invalid validation indices
        # -------------------------------------------------------------
        full_num_examples = len(self._image_list)
        validate_indices = [x for x in VALIDATE_INDICES if x in range(full_num_examples)]

        # ----------------------------------------------------------
        # Construct list of indices for training/validation
        # ----------------------------------------------------------
        list_of_indices = None
        if dstype == "train":
            list_of_indices = [x for x in range(full_num_examples) if x not in validate_indices]
        elif dstype == "valid":
            list_of_indices = validate_indices
        elif dstype == "full":
            list_of_indices = range(full_num_examples)
        else:
            raise ValueError("dstype '%s' unknown!", dstype)

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = [self._image_list[i] for i in list_of_indices]
        self._flow_list = [self._flow_list[i] for i in list_of_indices]
        self._occ_list = [self._occ_list[i] for i in list_of_indices]

        if imgtype is "comb":
            #image_list_final = [[val[0].replace("clean", "final"), val[1].replace("clean", "final")] for idx, val in enumerate(self._image_list)]
            image_list_final = [[val[i].replace("clean", "final") for i in range(len(val))] for idx, val in enumerate(self._image_list)]
            self._image_list += image_list_final
            self._flow_list += self._flow_list
            self._occ_list += self._occ_list

        assert len(self._image_list) == len(self._flow_list)
        assert len(self._image_list) == len(self._occ_list)

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

        self._size = len(self._image_list)

    def __getitem__(self, index):
        index = index % self._size

        imgs_filenames = self._image_list[index]
        flows_filenames = self._flow_list[index]
        occs_filenames = self._occ_list[index]

        #nframes = self._nframes
        #idx_first = len(imgs_filenames) - nframes

        # read float32 images and flow
        imgs_np0 = [common.read_image_as_byte(filename) for filename in imgs_filenames]
        flows_np0 = [common.read_flo_as_float32(filename) for filename in flows_filenames]
        occs_np0 = [common.read_occ_image_as_float32(filename) for filename in occs_filenames]

        # possibly apply photometric transformations
        imgs = self._photometric_transform(*imgs_np0)

        # convert flow to FloatTensor
        flows = [common.numpy2torch(flo_np0) for flo_np0 in flows_np0]

        # convert occ to FloatTensor
        occs = [common.numpy2torch(occ_np0) for occ_np0 in occs_np0]

        # e.g. "clean/alley_1/"
        basedir = os.path.splitext(os.path.dirname(imgs_filenames[0]).replace(self._substract_base, "")[1:])[0]

        # example filename
        basenames = [os.path.splitext(os.path.basename(f))[0] for f in imgs_filenames]

        example_dict = {
            "input1": imgs[0],
            "input_images": imgs, # "target_flows": torch.stack(flows, 0),
            "target1": flows[0],
            "target_flows": flows, #torch.stack(flows, 0)
            "target_occ1": occs[0],
            "target_occs": occs, #torch.stack(occs, 0)
            "index": index,
            "basedir": basedir,
            "basename": basenames,
            "nframes":self._nframes
        }

        return example_dict

    def __len__(self):
        return self._size



class _SintelMultiframe_test(data.Dataset):
    def __init__(self,
                 args,
                 dir_root=None, nframes=5,
                 photometric_augmentations=False,
                 imgtype=None,
                 reverse=False):

        self._args = args
        self._nframes = nframes
        self.reverse = reverse

        images_root = os.path.join(dir_root, imgtype)
        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")

        all_img_filenames = sorted(glob(os.path.join(images_root, "*/*.png")))

        # Remember base for substraction at runtime
        # e.g. subtract_base = "/home/user/.../MPI-Sintel-Complete/training/clean"
        self._substract_base = tools.cd_dotdot(images_root)

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = tools.cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))

        self._image_list = []

        for base_folder in base_folders:
            img_filenames = [x for x in all_img_filenames if base_folder in x]

            if self._nframes >= 0:
                for i in range(len(img_filenames) - self._nframes + 1):
                    imgs = img_filenames[i:i+self._nframes]
                    if self.reverse:
                        imgs = imgs[::-1]
                    self._image_list += [imgs]

                    # Sanity check
                    for k in range(len(imgs)-1):
                        im1_base_filename = os.path.splitext(os.path.basename(imgs[k]))[0]
                        im2_base_filename = os.path.splitext(os.path.basename(imgs[k+1]))[0]
                        im1_frame, im1_no = im1_base_filename.split("_")
                        im2_frame, im2_no = im2_base_filename.split("_")
                        assert(im1_frame == im2_frame)
                        if self.reverse:
                            assert(int(im2_no) == int(im1_no) - 1)
                        else:
                            assert(int(im1_no) == int(im2_no) - 1)
            else:
                imgs = img_filenames
                self._image_list += [imgs]

        # -------------------------------------------------------------
        # Remove invalid validation indices
        # -------------------------------------------------------------
        #import ipdb; ipdb.set_trace()
        full_num_examples = len(self._image_list)
        list_of_indices = range(full_num_examples)

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = [self._image_list[i] for i in list_of_indices]

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

        self._size = len(self._image_list)

    def __getitem__(self, index):
        index = index % self._size

        imgs_filenames = self._image_list[index]

        #nframes = self._nframes
        #idx_first = len(imgs_filenames) - nframes

        # read float32 images and flow
        imgs_np0 = [common.read_image_as_byte(filename) for filename in imgs_filenames]

        # possibly apply photometric transformations
        imgs = self._photometric_transform(*imgs_np0)

        # e.g. "clean/alley_1/"
        basedir = os.path.splitext(os.path.dirname(imgs_filenames[0]).replace(self._substract_base, "")[1:])[0]

        # example filename
        basenames = [os.path.splitext(os.path.basename(f))[0] for f in imgs_filenames]

        example_dict = {
            "input1": imgs[0],
            "input_images": imgs, # "target_flows": torch.stack(flows, 0),
            "index": index,
            "basedir": basedir,
            "basename": basenames,
            "nframes":self._nframes
        }

        return example_dict

    def __len__(self):
        return self._size



class SintelMultiframeTrainingCleanFull(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingCleanFull, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="clean",
            dstype="full", reverse=reverse)



class SintelMultiframeTrainingFinalFull(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingFinalFull, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="final",
            dstype="full", reverse=reverse)


class SintelMultiframeTrainingCombFull(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingCombFull, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="comb",
            dstype="full", reverse=reverse)



class SintelMultiframeTrainingCleanValid(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingCleanValid, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="clean",
            dstype="valid", reverse=reverse)



class SintelMultiframeTrainingFinalValid(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingFinalValid, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="final",
            dstype="valid", reverse=reverse)


class SintelMultiframeTrainingCombValid(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingCombValid, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="comb",
            dstype="valid", reverse=reverse)



class SintelMultiframeTrainingCleanTrain(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingCleanTrain, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="clean",
            dstype="train", reverse=reverse)



class SintelMultiframeTrainingFinalTrain(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingFinalTrain, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="final",
            dstype="train", reverse=reverse)


class SintelMultiframeTrainingCombTrain(_SintelMultiframe):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "training")
        super(SintelMultiframeTrainingCombTrain, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="comb",
            dstype="train", reverse=reverse)


class SintelMultiframeTestClean(_SintelMultiframe_test):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "test")
        super(SintelMultiframeTestClean, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="clean", reverse=reverse)

class SintelMultiframeTestFinal(_SintelMultiframe_test):
    def __init__(self, args, root, nframes=5, photometric_augmentations=True, reverse=False):
        dir_root = os.path.join(root, "test")
        super(SintelMultiframeTestFinal, self).__init__(
            args,
            dir_root=dir_root,
            nframes=nframes,
            photometric_augmentations=photometric_augmentations,
            imgtype="final", reverse=reverse)
