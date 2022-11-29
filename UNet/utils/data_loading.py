import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import skimage.io as skio
import skimage
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        self.img_ids = [splitext(file)[0] for file in sorted(listdir(images_dir)) if not file.startswith('.')]
        self.mask_ids = [splitext(file)[0] for file in sorted(listdir(masks_dir)) if not file.startswith('.')]
        if not self.img_ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        if len(self.mask_ids) != len(self.img_ids):
            raise RuntimeError(f'Wrong number of masks in {masks_dir}, make sure you put your masks there')
        logging.info(f'Creating dataset with {len(self.img_ids)} examples')

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def preprocess(img, is_mask):
        if is_mask :
            img_ndarray = np.asarray(img > 0)

        else :
            img_float = skimage.img_as_float(img)
            img_norm = (img_float -img_float.min())/(img_float.max() - img_float.min() + 1e-6)
            img_ndarray = np.asarray(img_norm)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        return img_ndarray

    @staticmethod
    def load(filename):
        return skio.imread(filename,plugin='pil')

    def __getitem__(self, idx):
        #print(f'Go load {idx}')
        name_img = self.img_ids[idx]
        name_mask = self.mask_ids[idx]
        #print(f'According to load idx {idx}, name_img = {name_img}')
        #print(f'According to load idx {idx}, name_mask = {name_mask}')
        mask_file = list(self.masks_dir.glob(name_mask + '.tif'))
        img_file = list(self.images_dir.glob(name_img + '.tif'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name_img}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name_mask}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name_img} & {name_mask} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }
