import torch
import os
import torchvision.transforms as transforms
from PIL import Image

import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

import exinput

class StyleDataset(torch.utils.data.Dataset):
    def __init__(self, content_dir):
        self.content = os.listdir(content_dir)
        self.content_dir = content_dir

        self.transform = transforms.Compose([
            transforms.Resize((512), antialias=True),
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        content = self.content[index]
        content = os.path.join(self.content_dir, content)
        content = Image.open(content).convert("RGB")
        content = self.transform(content)

        return content
    
    @staticmethod
    @pipeline_def(device_id=0, py_start_method="spawn")
    def dali_pipeline(content_dir, bs):
        content_images = fn.external_source(
            source=exinput.ExternalInputCallable(content_dir, bs), 
            parallel=True, 
            batch=False
        )
        
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
