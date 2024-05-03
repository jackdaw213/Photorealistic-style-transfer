import torch
import os
import torchvision.transforms as transforms
from cucim.skimage.color import rgb2lab
from PIL import Image
import torch.nn.functional as F

import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

import utils

class StyleDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
    
    @staticmethod
    @pipeline_def(device_id=0)
    def dali_pipeline(content_dir):
        content_images, _ = fn.readers.file(file_root=content_dir, 
                                            files=utils.list_images(content_dir),
                                            random_shuffle=True, 
                                            name="Reader")
        
        content_images = fn.decoders.image(content_images, device="mixed", output_type=types.RGB)

        content_images = fn.resize(content_images, size=512, dtype=types.FLOAT)

        content_images = fn.crop_mirror_normalize(content_images, 
                                                dtype=types.FLOAT,
                                                crop=(256, 256),
                                                crop_pos_x=fn.random.uniform(range=(0, 1)),
                                                crop_pos_y=fn.random.uniform(range=(0, 1)))
        
        """
        After training the photo-real style model the results are a bit odd. The details and transferred
        style look great but the images themself look over brighten. I thought it was because something
        was wrong in the model (there was but not related to this) or the loss function. Until I realized
        that ImageNet stats norm might be the cupid and I was right on the money. I removed it and the
        results are amazing, no more over-brightening. So I guess ImageNet stats norm is not suitable for
        everything. Man, the amount of stuff that can go wrong and screw the model up is insane, A may
        works fine with X but not with Y. And you only know it after you train the model and run it
        (there are probably some ways to detect these beforehand but I just don't know about them)
        """
        return content_images / 255
