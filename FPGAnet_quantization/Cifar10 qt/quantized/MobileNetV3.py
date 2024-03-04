# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class MobileNetV3(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(MobileNetV3, self).__init__()
        self.module_0 = py_nndct.nn.Input() #MobileNetV3::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ConvLayer[first_conv]/Conv2d[conv]/input.3
        self.module_2 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ConvLayer[first_conv]/Hardswish[act]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=24, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[0]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.9
        self.module_4 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[0]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.13
        self.module_5 = py_nndct.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[0]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.15
        self.module_6 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[0]/input.17
        self.module_7 = py_nndct.nn.Conv2d(in_channels=24, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[1]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.19
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[1]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/ReLU[act]/input.23
        self.module_9 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=96, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[1]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.25
        self.module_10 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[1]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.29
        self.module_11 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[1]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.31
        self.module_12 = py_nndct.nn.Conv2d(in_channels=32, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[2]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.35
        self.module_13 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[2]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/ReLU[act]/input.39
        self.module_14 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=96, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[2]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.41
        self.module_15 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[2]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.45
        self.module_16 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[2]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.47
        self.module_17 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[2]/input.49
        self.module_18 = py_nndct.nn.Conv2d(in_channels=32, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[3]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.51
        self.module_19 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[3]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/ReLU[act]/input.55
        self.module_20 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=96, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[3]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.57
        self.module_21 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[3]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.61
        self.module_22 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[3]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.63
        self.module_23 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[3]/input.65
        self.module_24 = py_nndct.nn.Conv2d(in_channels=32, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[4]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.67
        self.module_25 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[4]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/ReLU[act]/input.71
        self.module_26 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=96, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[4]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.73
        self.module_27 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[4]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.77
        self.module_28 = py_nndct.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[4]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.79
        self.module_29 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[4]/input.81
        self.module_30 = py_nndct.nn.Conv2d(in_channels=32, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.83
        self.module_31 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/ReLU[act]/input.87
        self.module_32 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=192, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.89
        self.module_33 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.93
        self.module_34 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.95
        self.module_35 = py_nndct.nn.Conv2d(in_channels=192, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.97
        self.module_36 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.99
        self.module_37 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.101
        self.module_38 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/14602
        self.module_39 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.103
        self.module_40 = py_nndct.nn.Conv2d(in_channels=192, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[5]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.105
        self.module_41 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.109
        self.module_42 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/ReLU[act]/input.113
        self.module_43 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=192, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.115
        self.module_44 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.119
        self.module_45 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.121
        self.module_46 = py_nndct.nn.Conv2d(in_channels=192, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.123
        self.module_47 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.125
        self.module_48 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.127
        self.module_49 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/14736
        self.module_50 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.129
        self.module_51 = py_nndct.nn.Conv2d(in_channels=192, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.131
        self.module_52 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[6]/input.133
        self.module_53 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.135
        self.module_54 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/ReLU[act]/input.139
        self.module_55 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=192, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.141
        self.module_56 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.145
        self.module_57 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.147
        self.module_58 = py_nndct.nn.Conv2d(in_channels=192, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.149
        self.module_59 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.151
        self.module_60 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.153
        self.module_61 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/14873
        self.module_62 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.155
        self.module_63 = py_nndct.nn.Conv2d(in_channels=192, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.157
        self.module_64 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[7]/input.159
        self.module_65 = py_nndct.nn.Conv2d(in_channels=48, out_channels=144, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.161
        self.module_66 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/ReLU[act]/input.165
        self.module_67 = py_nndct.nn.Conv2d(in_channels=144, out_channels=144, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=144, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.167
        self.module_68 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[depth_conv]/ReLU[act]/input.171
        self.module_69 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.173
        self.module_70 = py_nndct.nn.Conv2d(in_channels=144, out_channels=40, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.175
        self.module_71 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.177
        self.module_72 = py_nndct.nn.Conv2d(in_channels=40, out_channels=144, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.179
        self.module_73 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/15010
        self.module_74 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.181
        self.module_75 = py_nndct.nn.Conv2d(in_channels=144, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.183
        self.module_76 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[8]/input.185
        self.module_77 = py_nndct.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[9]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.187
        self.module_78 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[9]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.191
        self.module_79 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups=192, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[9]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.193
        self.module_80 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[9]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.197
        self.module_81 = py_nndct.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[9]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.199
        self.module_82 = py_nndct.nn.Conv2d(in_channels=96, out_channels=288, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[10]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.203
        self.module_83 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[10]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.207
        self.module_84 = py_nndct.nn.Conv2d(in_channels=288, out_channels=288, kernel_size=[7, 7], stride=[1, 1], padding=[3, 3], dilation=[1, 1], groups=288, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[10]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.209
        self.module_85 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[10]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.213
        self.module_86 = py_nndct.nn.Conv2d(in_channels=288, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[10]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.215
        self.module_87 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[10]/input.217
        self.module_88 = py_nndct.nn.Conv2d(in_channels=96, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[11]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.219
        self.module_89 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[11]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.223
        self.module_90 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=384, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[11]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.225
        self.module_91 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[11]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.229
        self.module_92 = py_nndct.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[11]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.231
        self.module_93 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[11]/input.233
        self.module_94 = py_nndct.nn.Conv2d(in_channels=96, out_channels=576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[12]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.235
        self.module_95 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[12]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.239
        self.module_96 = py_nndct.nn.Conv2d(in_channels=576, out_channels=576, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=576, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[12]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.241
        self.module_97 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[12]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.245
        self.module_98 = py_nndct.nn.Conv2d(in_channels=576, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[12]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.247
        self.module_99 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[12]/input.249
        self.module_100 = py_nndct.nn.Conv2d(in_channels=96, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.251
        self.module_101 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.255
        self.module_102 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=384, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.257
        self.module_103 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.261
        self.module_104 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.263
        self.module_105 = py_nndct.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.265
        self.module_106 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.267
        self.module_107 = py_nndct.nn.Conv2d(in_channels=96, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.269
        self.module_108 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/15464
        self.module_109 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.271
        self.module_110 = py_nndct.nn.Conv2d(in_channels=384, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[13]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.273
        self.module_111 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.277
        self.module_112 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.281
        self.module_113 = py_nndct.nn.Conv2d(in_channels=544, out_channels=544, kernel_size=[7, 7], stride=[1, 1], padding=[3, 3], dilation=[1, 1], groups=544, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.283
        self.module_114 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.287
        self.module_115 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.289
        self.module_116 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.291
        self.module_117 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.293
        self.module_118 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.295
        self.module_119 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/15598
        self.module_120 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.297
        self.module_121 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.299
        self.module_122 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[14]/input.301
        self.module_123 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.303
        self.module_124 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.307
        self.module_125 = py_nndct.nn.Conv2d(in_channels=544, out_channels=544, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=544, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.309
        self.module_126 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.313
        self.module_127 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.315
        self.module_128 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.317
        self.module_129 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.319
        self.module_130 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.321
        self.module_131 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/15735
        self.module_132 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.323
        self.module_133 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.325
        self.module_134 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[15]/input.327
        self.module_135 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.329
        self.module_136 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.333
        self.module_137 = py_nndct.nn.Conv2d(in_channels=544, out_channels=544, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=544, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.335
        self.module_138 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.339
        self.module_139 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.341
        self.module_140 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.343
        self.module_141 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.345
        self.module_142 = py_nndct.nn.Conv2d(in_channels=136, out_channels=544, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.347
        self.module_143 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/15872
        self.module_144 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.349
        self.module_145 = py_nndct.nn.Conv2d(in_channels=544, out_channels=136, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.351
        self.module_146 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[16]/input.353
        self.module_147 = py_nndct.nn.Conv2d(in_channels=136, out_channels=816, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.355
        self.module_148 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.359
        self.module_149 = py_nndct.nn.Conv2d(in_channels=816, out_channels=816, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=816, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.361
        self.module_150 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.365
        self.module_151 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.367
        self.module_152 = py_nndct.nn.Conv2d(in_channels=816, out_channels=208, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.369
        self.module_153 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.371
        self.module_154 = py_nndct.nn.Conv2d(in_channels=208, out_channels=816, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.373
        self.module_155 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/16009
        self.module_156 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.375
        self.module_157 = py_nndct.nn.Conv2d(in_channels=816, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[17]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.377
        self.module_158 = py_nndct.nn.Conv2d(in_channels=192, out_channels=1152, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.381
        self.module_159 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.385
        self.module_160 = py_nndct.nn.Conv2d(in_channels=1152, out_channels=1152, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1152, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.387
        self.module_161 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.391
        self.module_162 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.393
        self.module_163 = py_nndct.nn.Conv2d(in_channels=1152, out_channels=288, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.395
        self.module_164 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.397
        self.module_165 = py_nndct.nn.Conv2d(in_channels=288, out_channels=1152, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.399
        self.module_166 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/16143
        self.module_167 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.401
        self.module_168 = py_nndct.nn.Conv2d(in_channels=1152, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.403
        self.module_169 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[18]/input.405
        self.module_170 = py_nndct.nn.Conv2d(in_channels=192, out_channels=1152, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.407
        self.module_171 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.411
        self.module_172 = py_nndct.nn.Conv2d(in_channels=1152, out_channels=1152, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1152, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.413
        self.module_173 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.417
        self.module_174 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.419
        self.module_175 = py_nndct.nn.Conv2d(in_channels=1152, out_channels=288, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.421
        self.module_176 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.423
        self.module_177 = py_nndct.nn.Conv2d(in_channels=288, out_channels=1152, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.425
        self.module_178 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/16280
        self.module_179 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.427
        self.module_180 = py_nndct.nn.Conv2d(in_channels=1152, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.429
        self.module_181 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[19]/input.431
        self.module_182 = py_nndct.nn.Conv2d(in_channels=192, out_channels=576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Conv2d[conv]/input.433
        self.module_183 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[inverted_bottleneck]/Hardswish[act]/input.437
        self.module_184 = py_nndct.nn.Conv2d(in_channels=576, out_channels=576, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=576, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[depth_conv]/Conv2d[conv]/input.439
        self.module_185 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[depth_conv]/Hardswish[act]/input.443
        self.module_186 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/AdaptiveAvgPool2d[avgpool]/input.445
        self.module_187 = py_nndct.nn.Conv2d(in_channels=576, out_channels=144, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc1]/input.447
        self.module_188 = py_nndct.nn.ReLU(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/ReLU[activation]/input.449
        self.module_189 = py_nndct.nn.Conv2d(in_channels=144, out_channels=576, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Conv2d[fc2]/input.451
        self.module_190 = py_nndct.nn.Hardsigmoid(inplace=False) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/Hardsigmoid[scale_activation]/16417
        self.module_191 = py_nndct.nn.Module('nndct_elemwise_mul') #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[depth_conv]/SqueezeExcitation[se]/input.453
        self.module_192 = py_nndct.nn.Conv2d(in_channels=576, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/MBConvLayer[conv]/Sequential[point_linear]/Conv2d[conv]/input.455
        self.module_193 = py_nndct.nn.Add() #MobileNetV3::MobileNetV3/ResidualBlock[blocks]/ModuleList[20]/input.457
        self.module_194 = py_nndct.nn.Conv2d(in_channels=192, out_channels=1152, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MobileNetV3::MobileNetV3/ConvLayer[final_expand_layer]/Conv2d[conv]/input.459
        self.module_195 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ConvLayer[final_expand_layer]/Hardswish[act]/input.463
        self.module_196 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #MobileNetV3::MobileNetV3/AdaptiveAvgPool2d[global_avg_pool]/input.465
        self.module_197 = py_nndct.nn.Conv2d(in_channels=1152, out_channels=1536, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #MobileNetV3::MobileNetV3/ConvLayer[feature_mix_layer]/Conv2d[conv]/input
        self.module_198 = py_nndct.nn.Hardswish(inplace=False) #MobileNetV3::MobileNetV3/ConvLayer[feature_mix_layer]/Hardswish[act]/16509
        self.module_199 = py_nndct.nn.Module('nndct_shape') #MobileNetV3::MobileNetV3/16511
        self.module_200 = py_nndct.nn.Module('nndct_reshape') #MobileNetV3::MobileNetV3/16516
        self.module_201 = py_nndct.nn.Linear(in_features=1536, out_features=10, bias=True) #MobileNetV3::MobileNetV3/Linear[classifier]/16517

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_3 = self.module_3(output_module_0)
        output_module_3 = self.module_4(output_module_3)
        output_module_3 = self.module_5(output_module_3)
        output_module_3 = self.module_6(input=output_module_3, other=output_module_0, alpha=1)
        output_module_3 = self.module_7(output_module_3)
        output_module_3 = self.module_8(output_module_3)
        output_module_3 = self.module_9(output_module_3)
        output_module_3 = self.module_10(output_module_3)
        output_module_3 = self.module_11(output_module_3)
        output_module_12 = self.module_12(output_module_3)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_12 = self.module_15(output_module_12)
        output_module_12 = self.module_16(output_module_12)
        output_module_12 = self.module_17(input=output_module_12, other=output_module_3, alpha=1)
        output_module_18 = self.module_18(output_module_12)
        output_module_18 = self.module_19(output_module_18)
        output_module_18 = self.module_20(output_module_18)
        output_module_18 = self.module_21(output_module_18)
        output_module_18 = self.module_22(output_module_18)
        output_module_18 = self.module_23(input=output_module_18, other=output_module_12, alpha=1)
        output_module_24 = self.module_24(output_module_18)
        output_module_24 = self.module_25(output_module_24)
        output_module_24 = self.module_26(output_module_24)
        output_module_24 = self.module_27(output_module_24)
        output_module_24 = self.module_28(output_module_24)
        output_module_24 = self.module_29(input=output_module_24, other=output_module_18, alpha=1)
        output_module_24 = self.module_30(output_module_24)
        output_module_24 = self.module_31(output_module_24)
        output_module_24 = self.module_32(output_module_24)
        output_module_24 = self.module_33(output_module_24)
        output_module_34 = self.module_34(output_module_24)
        output_module_34 = self.module_35(output_module_34)
        output_module_34 = self.module_36(output_module_34)
        output_module_34 = self.module_37(output_module_34)
        output_module_34 = self.module_38(output_module_34)
        output_module_34 = self.module_39(input=output_module_34, other=output_module_24)
        output_module_34 = self.module_40(output_module_34)
        output_module_41 = self.module_41(output_module_34)
        output_module_41 = self.module_42(output_module_41)
        output_module_41 = self.module_43(output_module_41)
        output_module_41 = self.module_44(output_module_41)
        output_module_45 = self.module_45(output_module_41)
        output_module_45 = self.module_46(output_module_45)
        output_module_45 = self.module_47(output_module_45)
        output_module_45 = self.module_48(output_module_45)
        output_module_45 = self.module_49(output_module_45)
        output_module_45 = self.module_50(input=output_module_45, other=output_module_41)
        output_module_45 = self.module_51(output_module_45)
        output_module_45 = self.module_52(input=output_module_45, other=output_module_34, alpha=1)
        output_module_53 = self.module_53(output_module_45)
        output_module_53 = self.module_54(output_module_53)
        output_module_53 = self.module_55(output_module_53)
        output_module_53 = self.module_56(output_module_53)
        output_module_57 = self.module_57(output_module_53)
        output_module_57 = self.module_58(output_module_57)
        output_module_57 = self.module_59(output_module_57)
        output_module_57 = self.module_60(output_module_57)
        output_module_57 = self.module_61(output_module_57)
        output_module_57 = self.module_62(input=output_module_57, other=output_module_53)
        output_module_57 = self.module_63(output_module_57)
        output_module_57 = self.module_64(input=output_module_57, other=output_module_45, alpha=1)
        output_module_65 = self.module_65(output_module_57)
        output_module_65 = self.module_66(output_module_65)
        output_module_65 = self.module_67(output_module_65)
        output_module_65 = self.module_68(output_module_65)
        output_module_69 = self.module_69(output_module_65)
        output_module_69 = self.module_70(output_module_69)
        output_module_69 = self.module_71(output_module_69)
        output_module_69 = self.module_72(output_module_69)
        output_module_69 = self.module_73(output_module_69)
        output_module_69 = self.module_74(input=output_module_69, other=output_module_65)
        output_module_69 = self.module_75(output_module_69)
        output_module_69 = self.module_76(input=output_module_69, other=output_module_57, alpha=1)
        output_module_69 = self.module_77(output_module_69)
        output_module_69 = self.module_78(output_module_69)
        output_module_69 = self.module_79(output_module_69)
        output_module_69 = self.module_80(output_module_69)
        output_module_69 = self.module_81(output_module_69)
        output_module_82 = self.module_82(output_module_69)
        output_module_82 = self.module_83(output_module_82)
        output_module_82 = self.module_84(output_module_82)
        output_module_82 = self.module_85(output_module_82)
        output_module_82 = self.module_86(output_module_82)
        output_module_82 = self.module_87(input=output_module_82, other=output_module_69, alpha=1)
        output_module_88 = self.module_88(output_module_82)
        output_module_88 = self.module_89(output_module_88)
        output_module_88 = self.module_90(output_module_88)
        output_module_88 = self.module_91(output_module_88)
        output_module_88 = self.module_92(output_module_88)
        output_module_88 = self.module_93(input=output_module_88, other=output_module_82, alpha=1)
        output_module_94 = self.module_94(output_module_88)
        output_module_94 = self.module_95(output_module_94)
        output_module_94 = self.module_96(output_module_94)
        output_module_94 = self.module_97(output_module_94)
        output_module_94 = self.module_98(output_module_94)
        output_module_94 = self.module_99(input=output_module_94, other=output_module_88, alpha=1)
        output_module_94 = self.module_100(output_module_94)
        output_module_94 = self.module_101(output_module_94)
        output_module_94 = self.module_102(output_module_94)
        output_module_94 = self.module_103(output_module_94)
        output_module_104 = self.module_104(output_module_94)
        output_module_104 = self.module_105(output_module_104)
        output_module_104 = self.module_106(output_module_104)
        output_module_104 = self.module_107(output_module_104)
        output_module_104 = self.module_108(output_module_104)
        output_module_104 = self.module_109(input=output_module_104, other=output_module_94)
        output_module_104 = self.module_110(output_module_104)
        output_module_111 = self.module_111(output_module_104)
        output_module_111 = self.module_112(output_module_111)
        output_module_111 = self.module_113(output_module_111)
        output_module_111 = self.module_114(output_module_111)
        output_module_115 = self.module_115(output_module_111)
        output_module_115 = self.module_116(output_module_115)
        output_module_115 = self.module_117(output_module_115)
        output_module_115 = self.module_118(output_module_115)
        output_module_115 = self.module_119(output_module_115)
        output_module_115 = self.module_120(input=output_module_115, other=output_module_111)
        output_module_115 = self.module_121(output_module_115)
        output_module_115 = self.module_122(input=output_module_115, other=output_module_104, alpha=1)
        output_module_123 = self.module_123(output_module_115)
        output_module_123 = self.module_124(output_module_123)
        output_module_123 = self.module_125(output_module_123)
        output_module_123 = self.module_126(output_module_123)
        output_module_127 = self.module_127(output_module_123)
        output_module_127 = self.module_128(output_module_127)
        output_module_127 = self.module_129(output_module_127)
        output_module_127 = self.module_130(output_module_127)
        output_module_127 = self.module_131(output_module_127)
        output_module_127 = self.module_132(input=output_module_127, other=output_module_123)
        output_module_127 = self.module_133(output_module_127)
        output_module_127 = self.module_134(input=output_module_127, other=output_module_115, alpha=1)
        output_module_135 = self.module_135(output_module_127)
        output_module_135 = self.module_136(output_module_135)
        output_module_135 = self.module_137(output_module_135)
        output_module_135 = self.module_138(output_module_135)
        output_module_139 = self.module_139(output_module_135)
        output_module_139 = self.module_140(output_module_139)
        output_module_139 = self.module_141(output_module_139)
        output_module_139 = self.module_142(output_module_139)
        output_module_139 = self.module_143(output_module_139)
        output_module_139 = self.module_144(input=output_module_139, other=output_module_135)
        output_module_139 = self.module_145(output_module_139)
        output_module_139 = self.module_146(input=output_module_139, other=output_module_127, alpha=1)
        output_module_139 = self.module_147(output_module_139)
        output_module_139 = self.module_148(output_module_139)
        output_module_139 = self.module_149(output_module_139)
        output_module_139 = self.module_150(output_module_139)
        output_module_151 = self.module_151(output_module_139)
        output_module_151 = self.module_152(output_module_151)
        output_module_151 = self.module_153(output_module_151)
        output_module_151 = self.module_154(output_module_151)
        output_module_151 = self.module_155(output_module_151)
        output_module_151 = self.module_156(input=output_module_151, other=output_module_139)
        output_module_151 = self.module_157(output_module_151)
        output_module_158 = self.module_158(output_module_151)
        output_module_158 = self.module_159(output_module_158)
        output_module_158 = self.module_160(output_module_158)
        output_module_158 = self.module_161(output_module_158)
        output_module_162 = self.module_162(output_module_158)
        output_module_162 = self.module_163(output_module_162)
        output_module_162 = self.module_164(output_module_162)
        output_module_162 = self.module_165(output_module_162)
        output_module_162 = self.module_166(output_module_162)
        output_module_162 = self.module_167(input=output_module_162, other=output_module_158)
        output_module_162 = self.module_168(output_module_162)
        output_module_162 = self.module_169(input=output_module_162, other=output_module_151, alpha=1)
        output_module_170 = self.module_170(output_module_162)
        output_module_170 = self.module_171(output_module_170)
        output_module_170 = self.module_172(output_module_170)
        output_module_170 = self.module_173(output_module_170)
        output_module_174 = self.module_174(output_module_170)
        output_module_174 = self.module_175(output_module_174)
        output_module_174 = self.module_176(output_module_174)
        output_module_174 = self.module_177(output_module_174)
        output_module_174 = self.module_178(output_module_174)
        output_module_174 = self.module_179(input=output_module_174, other=output_module_170)
        output_module_174 = self.module_180(output_module_174)
        output_module_174 = self.module_181(input=output_module_174, other=output_module_162, alpha=1)
        output_module_182 = self.module_182(output_module_174)
        output_module_182 = self.module_183(output_module_182)
        output_module_182 = self.module_184(output_module_182)
        output_module_182 = self.module_185(output_module_182)
        output_module_186 = self.module_186(output_module_182)
        output_module_186 = self.module_187(output_module_186)
        output_module_186 = self.module_188(output_module_186)
        output_module_186 = self.module_189(output_module_186)
        output_module_186 = self.module_190(output_module_186)
        output_module_186 = self.module_191(input=output_module_186, other=output_module_182)
        output_module_186 = self.module_192(output_module_186)
        output_module_186 = self.module_193(input=output_module_186, other=output_module_174, alpha=1)
        output_module_186 = self.module_194(output_module_186)
        output_module_186 = self.module_195(output_module_186)
        output_module_186 = self.module_196(output_module_186)
        output_module_186 = self.module_197(output_module_186)
        output_module_186 = self.module_198(output_module_186)
        output_module_199 = self.module_199(input=output_module_186, dim=0)
        output_module_200 = self.module_200(input=output_module_186, shape=[output_module_199,-1])
        output_module_200 = self.module_201(output_module_200)
        return output_module_200
