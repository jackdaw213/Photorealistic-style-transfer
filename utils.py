import os
import sys
import random

import torch

import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.color import rgb2lab

import numpy as np
from PIL import Image
import PIL
import torchvision.transforms.functional as F

def image_grid(**kwargs):
    col_names = list(kwargs.keys())
    num_rows = len(kwargs[col_names[0]])
    num_cols = len(col_names)
        
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 9))

    if num_rows != 1:
        for row in range(num_rows):
            for col in range(num_cols):
                ax = axs[row, col]
                if row == 0:
                    ax.set_title(col_names[col])
                ax.imshow(kwargs[col_names[col]][row])
                ax.axis('off')
    else:
        for col in range(num_cols):
            ax = axs[col]
            ax.set_title(col_names[col])
            ax.imshow(kwargs[col_names[col]][0])
            ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def test_style_model(model, con_images_path, sty_images_path, num_samples=8):
    cons = os.listdir(con_images_path)
    cons = np.random.choice(cons, num_samples, replace=False)

    stys = os.listdir(sty_images_path)
    stys = np.random.choice(stys, num_samples, replace=False)

    con_images = []
    sty_images = []
    output = []

    for con, sty in zip(cons, stys):   
        con = os.path.join(con_images_path, con) 
        con = Image.open(con).convert("RGB")
        con_images.append(con)
        con = F.to_tensor(con)

        sty = os.path.join(sty_images_path, sty) 
        sty = Image.open(sty).convert("RGB")
        sty_images.append(sty)
        sty = F.to_tensor(sty)

        con = con.unsqueeze(0).float()
        sty = sty.unsqueeze(0).float()

        model.eval()
        with torch.no_grad():
            out = model(con, sty)
        out = out.squeeze()
        output.append(F.to_pil_image(out))

    image_grid(Content=con_images, Style=sty_images, Output=output)            

def save_train_state(model, optimizer, scaler, epoch, path):
    # This one is for resuming training
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'epoch': epoch
    }, path)

    path = path + ".epoch" + str(epoch + 1)
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'epoch': epoch
    }, path)

def load_train_state(path):
    try:
        state = torch.load(path)
        return state["model"], state["optimizer"], state["scaler"], state["epoch"]
    except Exception as e:
        print(e)
        sys.exit("Loading train state failed, existing")

def pad_fetures(up, con_channels):
    """
    We need to pad the features with 0 when we concatenating upscaled 
    features that were previously downscaled from odd dimension features
    For example: 25 -> down -> 12 -> up -> 24 -> pad -> 25
    """
    diffY = con_channels.size()[2] - up.size()[2]
    diffX = con_channels.size()[3] - up.size()[3]
    up = torch.nn.functional.pad(up, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
    return up

def list_images(folder_path):
    """
    Instead of creating a labels file, we can just pass a list of files to the 
    decoder via files argument. And it does not take too much time either (2s 
    from my testing)
    """
    temp = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            temp.append(filename)
    return temp
