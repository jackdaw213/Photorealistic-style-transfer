import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class WCT(nn.Module):
    def __init__(self):
        super().__init__()

    def whiten_and_color(self, cF,sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF,1) # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
        c_u,c_e,c_v = torch.svd(contentConv,some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF,1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
        s_u,s_e,s_v = torch.svd(styleConv,some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
        step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
        whiten_cF = torch.mm(step2,cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def forward(self, cF,sF,alpha=1):
        cF = cF.double()
        sF = sF.double()
        if len(cF.size()) == 4:
            cF = cF[0]
        if len(sF.size()) == 4:
            sF = sF[0]
        C,W,H = cF.size(0),cF.size(1),cF.size(2)
        _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
        cFView = cF.view(C,-1)
        sFView = sF.view(C,-1)

        targetFeature = self.whiten_and_color(cFView,sFView)
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF
        csF = csF.float().unsqueeze(0)
        return csF
    
class StyleLoss(nn.Module):
    def __init__(self, _lambda=0.5):
        super().__init__()
        self._lambda = _lambda

    def contentLoss(self, content, content_out):
        return F.mse_loss(content, content_out)

    def styleLoss(self, content_features, content_features_loss):
        style_loss = 0

        for c, cl in zip(content_features, content_features_loss):
            style_loss += F.mse_loss(c, cl)

        return style_loss

    def forward(self, content, content_out, content_features, content_features_loss):
        return self.contentLoss(content, content_out) * (1 - self._lambda), self._lambda * self.styleLoss(content_features, content_features_loss)

class VggDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer):
        super().__init__()

        self.up_scale = nn.Upsample(scale_factor=2, mode='nearest')
        
        if layer < 3:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU() 
            )
        elif layer < 5:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU()
            )
        else:
            self.seq = nn.Sequential(
                # BFA
                nn.Conv2d(in_channels + 960, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.ReLU()
            )

    def forward(self, inp, con_channels):
        if con_channels is not None:
            inp = self.up_scale(inp)
            inp = utils.pad_fetures(inp, con_channels)
            inp = torch.cat([inp, con_channels], dim=1)
        return self.seq(inp)
