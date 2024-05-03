import torch
import torchvision.transforms.functional as F
import torch.nn.functional as f
import matplotlib.pyplot as plt
import argparse

from PIL import Image

import model
import utils

parser = argparse.ArgumentParser(description='Photorealistic style transfer')

parser.add_argument('-c', '--content', type=str,
                    help='Path to the content image')
parser.add_argument('-s', '--style', type=str,
                    help='Path to the style image')
parser.add_argument('-m', '--model', type=str,
                    help='Path to the model')

args = parser.parse_args()

style = model.StyleTransfer()
style.load_state_dict(torch.load("model/train.state", 
map_location=torch.device('cpu'))["model"])
style.eval()

content = F.to_tensor(Image.open(args.content).convert("RGB"))
style = F.to_tensor(Image.open(args.style).convert("RGB"))

with torch.no_grad():
    output = style(content, style)

output = output.squeeze()

plt.imshow(output.permute(1, 2, 0))
plt.show()
