import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models import VGG19_Weights
import model_parts as mp
import torch.nn.functional as F

class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()

        vgg19 = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features

        for param in vgg19.parameters():
            param.requires_grad = False

        # ReLU 1_1 -> ReLu 5_1
        self.e1 = nn.Sequential(*list(vgg19.children())[:2])
        self.e2 = nn.Sequential(*list(vgg19.children())[2:7])
        self.e3 = nn.Sequential(*list(vgg19.children())[7:12])
        self.e4 = nn.Sequential(*list(vgg19.children())[12:21])
        self.e5 = nn.Sequential(*list(vgg19.children())[21:30])

        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(128)
        self.in3 = nn.InstanceNorm2d(256)
        self.in4 = nn.InstanceNorm2d(512)

        self.max1 = nn.MaxPool2d(kernel_size=16,stride=16)
        self.max2 = nn.MaxPool2d(kernel_size=8,stride=8)
        self.max3 = nn.MaxPool2d(kernel_size=4,stride=4)
        self.max4 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.wct = mp.WCT()

        self.d5 = mp.VggDecoderBlock(512, 512, 5)
        self.d4 = mp.VggDecoderBlock(512, 256, 4)
        self.d3 = mp.VggDecoderBlock(256, 128, 3)
        self.d2 = mp.VggDecoderBlock(128, 64, 2)
        self.d1 = mp.VggDecoderBlock(64, 3, 1)


    def encoder(self, input):
        features = []

        out = self.e1(input)
        features.append(out)

        out = self.e2(out)
        features.append(out)

        out = self.e3(out)
        features.append(out)

        out = self.e4(out)
        features.append(out)

        out = self.e5(out)

        return out, features
    
    def bfa(self, input, concat):

        # They use instance norm before concatenating in the code base even though
        # this was not mentioned in the paper
        input = torch.concat([input, self.max1(self.in1(concat[0]))], dim=1)
        input = torch.concat([input, self.max2(self.in2(concat[1]))], dim=1)
        input = torch.concat([input, self.max3(self.in3(concat[2]))], dim=1)
        input = torch.concat([input, self.max4(self.in4(concat[3]))], dim=1)

        return input
    
    def insl(self, content_features, style_features):
        insl_features = []

        insl_features.append(self.wct(content_features[0], style_features[0]))
        insl_features.append(self.wct(content_features[1], style_features[1]))
        insl_features.append(self.wct(content_features[2], style_features[2]))
        insl_features.append(self.wct(content_features[3], style_features[3]))
        
        return insl_features

    def forward(self, content, style=None):
        # Training mode, no wct
        if style is None:
            content, content_features = self.encoder(content)
            content_features.append(content)

            content = self.bfa(content, content_features)

            content = self.d5(content, None)
            content = self.d4(content, self.in4(content_features[3]))
            content = self.d3(content, self.in3(content_features[2]))
            content = self.d2(content, self.in2(content_features[1]))
            content = self.d1(content, self.in1(content_features[0]))

            _, content_features_loss = self.encoder(content)
            content_features_loss.append(_)

            return content, content_features, content_features_loss
        
        content, content_features = self.encoder(content)
        style, style_features = self.encoder(style)

        insl = self.insl(content_features, style_features)
        content_wct = self.wct(content, style)

        content_wct = self.bfa(content_wct, insl)
        style = self.bfa(style, style_features)

        content_wct = self.d5(content_wct, None)
        style = self.d5(style, None)
        content_wct = self.wct(content_wct, style)

        content_wct = self.d4(content_wct, self.in4(insl[3]))
        style = self.d4(style, self.in4(style_features[3]))
        content_wct = self.wct(content_wct, style)

        content_wct = self.d3(content_wct, self.in3(insl[2]))
        style = self.d3(style, self.in3(style_features[2]))
        content_wct = self.wct(content_wct, style)

        content_wct = self.d2(content_wct, self.in2(insl[1]))
        style = self.d2(style, self.in2(style_features[1]))
        content_wct = self.wct(content_wct, style)

        content_wct = self.d1(content_wct, self.in1(insl[0]))

        return content_wct