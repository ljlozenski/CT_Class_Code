import torch
import numpy as np


class CombinedConvLayer(torch.nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size = 3, padding = 1, stride = 1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_feat, out_feat, kernel_size, padding = padding, stride = stride)
        self.bn = torch.nn.BatchNorm2d(out_feat)
        self.act = torch.nn.SiLU()
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))


class ImageUNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        nc0 = 16
        nc1 = 32
        nc2 = 64
        nc3 = 128
        nc4 = 256

        self.layer_00 = CombinedConvLayer(1,nc0)
        self.layer_01 = CombinedConvLayer(nc0,nc0)
        self.layer_02 = CombinedConvLayer(nc0,nc0)

        self.mp = torch.nn.MaxPool2d(2)
        self.up = torch.nn.Upsample(scale_factor = 2)

        self.layer_10 = CombinedConvLayer(nc0,nc1)
        self.layer_11 = CombinedConvLayer(nc1,nc1)
        self.layer_12 = CombinedConvLayer(nc1,nc1)

        self.layer_20 = CombinedConvLayer(nc1,nc2)
        self.layer_21 = CombinedConvLayer(nc2,nc2)
        self.layer_22 = CombinedConvLayer(nc2,nc2)

        self.layer_30 = CombinedConvLayer(nc2,nc3)
        self.layer_31 = CombinedConvLayer(nc3,nc3)
        self.layer_32 = CombinedConvLayer(nc3,nc3)

        self.layer_40 = CombinedConvLayer(nc3,nc4)
        self.layer_41 = CombinedConvLayer(nc4,nc4)
        self.layer_42 = CombinedConvLayer(nc4,nc4)

        self.layer_33 = CombinedConvLayer(nc4+nc3,nc3)
        self.layer_34 = CombinedConvLayer(nc3,nc3)
        self.layer_35 = CombinedConvLayer(nc3,nc3)

        self.layer_23 = CombinedConvLayer(nc3 + nc2,nc2)
        self.layer_24 = CombinedConvLayer(nc2,nc2)
        self.layer_25 = CombinedConvLayer(nc2,nc2)

        self.layer_13 = CombinedConvLayer(nc2 +nc1,nc1)
        self.layer_14 = CombinedConvLayer(nc1,nc1)
        self.layer_15 = CombinedConvLayer(nc1,nc1)

        self.layer_03 = CombinedConvLayer(nc0 + nc1,nc0)
        self.layer_04 = CombinedConvLayer(nc0,nc0)
        self.layer_05 = torch.nn.Conv2d(nc0,1, 3, padding = 1)

    def forward(self,x0):

        x0 = self.layer_00(x0)
        x0 = self.layer_01(x0)
        x0 = self.layer_02(x0)

        x1 = self.mp(x0)
        x1 = self.layer_10(x1)
        x1 = self.layer_11(x1)
        x1 = self.layer_12(x1)

        x2 = self.mp(x1)
        x2 = self.layer_20(x2)
        x2 = self.layer_21(x2)
        x2 = self.layer_22(x2)

        x3 = self.mp(x2)
        x3 = self.layer_30(x3)
        x3 = self.layer_31(x3)
        x3 = self.layer_32(x3)

        x4 = self.mp(x3)
        x4 = self.layer_40(x4)
        x4 = self.layer_41(x4)
        x4 = self.layer_42(x4)
        x4 = self.up(x4)

        x3 = torch.cat((x3, x4), dim = 1)
        x3 = self.layer_33(x3)
        x3 = self.layer_34(x3)
        x3 = self.layer_35(x3)
        x3 = self.up(x3)

        x2 = torch.cat((x2, x3), dim = 1)
        x2 = self.layer_23(x2)
        x2 = self.layer_24(x2)
        x2 = self.layer_25(x2)
        x2 = self.up(x2)

        x1 = torch.cat((x1, x2), dim = 1)
        x1 = self.layer_13(x1)
        x1 = self.layer_14(x1)
        x1 = self.layer_15(x1)
        x1 = self.up(x1)

        x0 = torch.cat((x0, x1), dim = 1)
        x0 = self.layer_03(x0)
        x0 = self.layer_04(x0)
        x0 = self.layer_05(x0)

        return x0

class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = ImageUNet()
    def forward(self,x):
        return x - 0.2*0.25*self.unet(x)

class MeasurementEncoder(torch.nn.Module):
    def __init__(self, nc0 = 16, nc1 = 32, nc2 = 64, nc3 = 128, nc4 = 256):
        super().__init__()

        self.layer_00 = CombinedConvLayer(1,nc0, kernel_size=(3,5),padding = 0, stride = (1,2))
        self.layer_01 = CombinedConvLayer(nc0,nc0, padding = 0)
        self.layer_02 = CombinedConvLayer(nc0,nc0, padding = 0)

        #self.mp = torch.nn.MaxPool2d(2)
        #self.up = torch.nn.Upsample(scale_factor = 2)

        self.layer_10 = CombinedConvLayer(nc0,nc1, kernel_size=(3,5),padding = 0, stride = (1,2))
        self.layer_11 = CombinedConvLayer(nc1,nc1,padding = 0)
        self.layer_12 = CombinedConvLayer(nc1,nc1,padding = 0)

        self.layer_20 = CombinedConvLayer(nc1,nc2, kernel_size=(3,5),padding = 0, stride = (1,2))
        self.layer_21 = CombinedConvLayer(nc2,nc2,kernel_size= 5, padding = 0)
        self.layer_22 = CombinedConvLayer(nc2,nc2,padding = 0)

        self.layer_30 = CombinedConvLayer(nc2,nc3, kernel_size=(3,5),padding = 0, stride = (1,1))
        self.layer_31 = CombinedConvLayer(nc3,nc3, padding = 0)
        self.layer_32 = CombinedConvLayer(nc3,nc3,padding = 0)

        self.layer_40 = CombinedConvLayer(nc3,nc4, kernel_size=(5,5),padding = 0, stride = (1,2))
        self.layer_41 = CombinedConvLayer(nc4,nc4,padding = 0)
        self.layer_42 = CombinedConvLayer(nc4,nc4,kernel_size=(4,2),padding = 0)
    def forward(self,x):
        x = self.layer_00(x)
        x = self.layer_01(x)
        x = self.layer_02(x)
        x = self.layer_10(x)
        x = self.layer_11(x)
        x = self.layer_12(x)
        x = self.layer_20(x)
        x = self.layer_21(x)
        x = self.layer_22(x)
        x = self.layer_30(x)
        x = self.layer_31(x)
        x = self.layer_32(x)
        x = self.layer_40(x)
        x = self.layer_41(x)
        x = self.layer_42(x)
        return x

class ImageDecoder(torch.nn.Module):
    def __init__(self, nc0 = 16, nc1 = 32, nc2 = 64, nc3 = 128, nc4 = 256):
        super().__init__()
        self.up2 = torch.nn.Upsample(scale_factor = 2)
        self.up4 = torch.nn.Upsample(scale_factor = 4)

        self.layer_00 = CombinedConvLayer(nc4,nc4)
        self.layer_01 = CombinedConvLayer(nc4,nc4)
        self.layer_02 = CombinedConvLayer(nc4,nc4)

        self.layer_10 = CombinedConvLayer(nc4,nc3)
        self.layer_11 = CombinedConvLayer(nc3,nc3)
        self.layer_12 = CombinedConvLayer(nc3,nc3)

        self.layer_20 = CombinedConvLayer(nc3,nc2)
        self.layer_21 = CombinedConvLayer(nc2,nc2)
        self.layer_22 = CombinedConvLayer(nc2,nc2)

        self.layer_30 = CombinedConvLayer(nc2,nc1)
        self.layer_31 = CombinedConvLayer(nc1,nc1)
        self.layer_32 = CombinedConvLayer(nc1,nc1)

        self.layer_40 = CombinedConvLayer(nc1,nc0)
        self.layer_41 = CombinedConvLayer(nc0,nc0)
        self.layer_42 = CombinedConvLayer(nc0,1)
    def forward(self,x):
        x = self.up4(x)
        x = self.layer_00(x)
        x = self.layer_01(x)
        x = self.layer_02(x)

        x = self.up4(x)
        x = self.layer_10(x)
        x = self.layer_11(x)
        x = self.layer_12(x)

        x = self.up2(x)
        x = self.layer_20(x)
        x = self.layer_21(x)
        x = self.layer_22(x)

        x = self.up4(x)
        x = self.layer_30(x)
        x = self.layer_31(x)
        x = self.layer_32(x)

        x = self.up2(x)
        x = self.layer_40(x)
        x = self.layer_41(x)
        x = self.layer_42(x)
        
        return x

class LearndInversion(torch.nn.Module):
    def __init__(self, nc0 = 16, nc1 = 32, nc2 = 64, nc3 = 128, nc4 = 256):
        super().__init__()
        self.encoder = MeasurementEncoder(nc0 = nc0, nc1 = nc1, nc2 = nc2, nc3 = nc3, nc4 = nc4)
        self.decoer = ImageDecoder(nc0 = nc0, nc1 = nc1, nc2 = nc2, nc3 = nc3, nc4 = nc4)
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoer(x)
        return x 

class DataUNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        nc0 = 16
        nc1 = 32
        nc2 = 64
        nc3 = 128
        nc4 = 256

        self.layer_00 = CombinedConvLayer(1,nc0)
        self.layer_01 = CombinedConvLayer(nc0,nc0)
        self.layer_02 = CombinedConvLayer(nc0,nc0)

        self.mp = torch.nn.MaxPool2d(2)
        self.up = torch.nn.Upsample(scale_factor = 2)


        self.mp2 = torch.nn.MaxPool2d((3,2))
        self.up2 = torch.nn.Upsample(scale_factor = (3,2))

        self.mp3 = torch.nn.MaxPool2d((1,2))
        self.up3 = torch.nn.Upsample(scale_factor = (1,2))

        self.layer_10 = CombinedConvLayer(nc0,nc1)
        self.layer_11 = CombinedConvLayer(nc1,nc1)
        self.layer_12 = CombinedConvLayer(nc1,nc1)

        self.layer_20 = CombinedConvLayer(nc1,nc2)
        self.layer_21 = CombinedConvLayer(nc2,nc2)
        self.layer_22 = CombinedConvLayer(nc2,nc2)

        self.layer_30 = CombinedConvLayer(nc2,nc3)
        self.layer_31 = CombinedConvLayer(nc3,nc3)
        self.layer_32 = CombinedConvLayer(nc3,nc3)

        self.layer_40 = CombinedConvLayer(nc3,nc4)
        self.layer_41 = CombinedConvLayer(nc4,nc4)
        self.layer_42 = CombinedConvLayer(nc4,nc4)

        self.layer_33 = CombinedConvLayer(nc4+nc3,nc3)
        self.layer_34 = CombinedConvLayer(nc3,nc3)
        self.layer_35 = CombinedConvLayer(nc3,nc3)

        self.layer_23 = CombinedConvLayer(nc3 + nc2,nc2)
        self.layer_24 = CombinedConvLayer(nc2,nc2)
        self.layer_25 = CombinedConvLayer(nc2,nc2)

        self.layer_13 = CombinedConvLayer(nc2 +nc1,nc1)
        self.layer_14 = CombinedConvLayer(nc1,nc1)
        self.layer_15 = CombinedConvLayer(nc1,nc1)

        self.layer_03 = CombinedConvLayer(nc0 + nc1,nc0)
        self.layer_04 = CombinedConvLayer(nc0,nc0)
        self.layer_05 = torch.nn.Conv2d(nc0,1, 3, padding = 1)

    def forward(self,x0):

        x0 = self.layer_00(x0)
        x0 = self.layer_01(x0)
        x0 = self.layer_02(x0)

        x1 = self.mp(x0)
        x1 = self.layer_10(x1)
        x1 = self.layer_11(x1)
        x1 = self.layer_12(x1)

        x2 = self.mp(x1)
        x2 = self.layer_20(x2)
        x2 = self.layer_21(x2)
        x2 = self.layer_22(x2)

        x3 = self.mp2(x2)
        x3 = self.layer_30(x3)
        x3 = self.layer_31(x3)
        x3 = self.layer_32(x3)

        x4 = self.mp3(x3)
        x4 = self.layer_40(x4)
        x4 = self.layer_41(x4)
        x4 = self.layer_42(x4)
        x4 = self.up3(x4)

        x3 = torch.cat((x3, x4), dim = 1)
        x3 = self.layer_33(x3)
        x3 = self.layer_34(x3)
        x3 = self.layer_35(x3)
        x3 = self.up2(x3)

        x2 = torch.cat((x2, x3), dim = 1)
        x2 = self.layer_23(x2)
        x2 = self.layer_24(x2)
        x2 = self.layer_25(x2)
        x2 = self.up(x2)

        x1 = torch.cat((x1, x2), dim = 1)
        x1 = self.layer_13(x1)
        x1 = self.layer_14(x1)
        x1 = self.layer_15(x1)
        x1 = self.up(x1)

        x0 = torch.cat((x0, x1), dim = 1)
        x0 = self.layer_03(x0)
        x0 = self.layer_04(x0)
        x0 = self.layer_05(x0)

        return x0

