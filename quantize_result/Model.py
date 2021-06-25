# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Model::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.2
        self.module_3 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.4
        self.module_4 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.5
        self.module_6 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.7
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.8
        self.module_9 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.10
        self.module_10 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.11
        self.module_12 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/512
        self.module_13 = py_nndct.nn.Add() #Model::Model/Bottleneck/input.13
        self.module_14 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.14
        self.module_16 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.16
        self.module_17 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.17
        self.module_19 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.19
        self.module_20 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.20
        self.module_22 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/568
        self.module_23 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[0]/input.22
        self.module_24 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.23
        self.module_26 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.25
        self.module_27 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.26
        self.module_29 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/606
        self.module_30 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[1]/input.28
        self.module_31 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.29
        self.module_33 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.31
        self.module_34 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.32
        self.module_36 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.34
        self.module_37 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.35
        self.module_39 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/662
        self.module_40 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[0]/input.37
        self.module_41 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.38
        self.module_43 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.40
        self.module_44 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.41
        self.module_46 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/700
        self.module_47 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[1]/input.43
        self.module_48 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.44
        self.module_50 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.46
        self.module_51 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.47
        self.module_53 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/738
        self.module_54 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[2]/input.49
        self.module_55 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.50
        self.module_57 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.52
        self.module_58 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.53
        self.module_60 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/776
        self.module_61 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[3]/input.55
        self.module_62 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[4]/Conv[cv1]/Conv2d[conv]/input.56
        self.module_64 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[4]/Conv[cv1]/LeakyReLU[act]/input.58
        self.module_65 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[4]/Conv[cv2]/Conv2d[conv]/input.59
        self.module_67 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[4]/Conv[cv2]/LeakyReLU[act]/814
        self.module_68 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[4]/input.61
        self.module_69 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[5]/Conv[cv1]/Conv2d[conv]/input.62
        self.module_71 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[5]/Conv[cv1]/LeakyReLU[act]/input.64
        self.module_72 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[5]/Conv[cv2]/Conv2d[conv]/input.65
        self.module_74 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[5]/Conv[cv2]/LeakyReLU[act]/852
        self.module_75 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[5]/input.67
        self.module_76 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[6]/Conv[cv1]/Conv2d[conv]/input.68
        self.module_78 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[6]/Conv[cv1]/LeakyReLU[act]/input.70
        self.module_79 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[6]/Conv[cv2]/Conv2d[conv]/input.71
        self.module_81 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[6]/Conv[cv2]/LeakyReLU[act]/890
        self.module_82 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[6]/input.73
        self.module_83 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[7]/Conv[cv1]/Conv2d[conv]/input.74
        self.module_85 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[7]/Conv[cv1]/LeakyReLU[act]/input.76
        self.module_86 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[7]/Conv[cv2]/Conv2d[conv]/input.77
        self.module_88 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[7]/Conv[cv2]/LeakyReLU[act]/928
        self.module_89 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[7]/input.79
        self.module_90 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.80
        self.module_92 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.82
        self.module_93 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.83
        self.module_95 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.85
        self.module_96 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.86
        self.module_98 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/984
        self.module_99 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[0]/input.88
        self.module_100 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.89
        self.module_102 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.91
        self.module_103 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.92
        self.module_105 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/1022
        self.module_106 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[1]/input.94
        self.module_107 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.95
        self.module_109 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.97
        self.module_110 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.98
        self.module_112 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/1060
        self.module_113 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[2]/input.100
        self.module_114 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.101
        self.module_116 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.103
        self.module_117 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.104
        self.module_119 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/1098
        self.module_120 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[3]/input.106
        self.module_121 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[4]/Conv[cv1]/Conv2d[conv]/input.107
        self.module_123 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[4]/Conv[cv1]/LeakyReLU[act]/input.109
        self.module_124 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[4]/Conv[cv2]/Conv2d[conv]/input.110
        self.module_126 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[4]/Conv[cv2]/LeakyReLU[act]/1136
        self.module_127 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[4]/input.112
        self.module_128 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[5]/Conv[cv1]/Conv2d[conv]/input.113
        self.module_130 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[5]/Conv[cv1]/LeakyReLU[act]/input.115
        self.module_131 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[5]/Conv[cv2]/Conv2d[conv]/input.116
        self.module_133 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[5]/Conv[cv2]/LeakyReLU[act]/1174
        self.module_134 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[5]/input.118
        self.module_135 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[6]/Conv[cv1]/Conv2d[conv]/input.119
        self.module_137 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[6]/Conv[cv1]/LeakyReLU[act]/input.121
        self.module_138 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[6]/Conv[cv2]/Conv2d[conv]/input.122
        self.module_140 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[6]/Conv[cv2]/LeakyReLU[act]/1212
        self.module_141 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[6]/input.124
        self.module_142 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[7]/Conv[cv1]/Conv2d[conv]/input.125
        self.module_144 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[7]/Conv[cv1]/LeakyReLU[act]/input.127
        self.module_145 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[7]/Conv[cv2]/Conv2d[conv]/input.128
        self.module_147 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[7]/Conv[cv2]/LeakyReLU[act]/1250
        self.module_148 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[7]/input.130
        self.module_149 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.131
        self.module_151 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.133
        self.module_152 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.134
        self.module_154 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.136
        self.module_155 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.137
        self.module_157 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/1306
        self.module_158 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[0]/input.139
        self.module_159 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.140
        self.module_161 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.142
        self.module_162 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.143
        self.module_164 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/1344
        self.module_165 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[1]/input.145
        self.module_166 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.146
        self.module_168 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.148
        self.module_169 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.149
        self.module_171 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/1382
        self.module_172 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[2]/input.151
        self.module_173 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.152
        self.module_175 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.154
        self.module_176 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.155
        self.module_178 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/1420
        self.module_179 = py_nndct.nn.Add() #Model::Model/Sequential/Bottleneck[3]/input.157
        self.module_180 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.158
        self.module_182 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.160
        self.module_183 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.161
        self.module_185 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/input.163
        self.module_186 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.164
        self.module_188 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.166
        self.module_189 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.167
        self.module_191 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.169
        self.module_192 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.170
        self.module_194 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.172
        self.module_195 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.173
        self.module_197 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input
        self.module_198 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.175
        self.module_200 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.177
        self.module_201 = py_nndct.nn.Module('shape') #Model::Model/Upsample/1550
        self.module_202 = py_nndct.nn.Module('tensor') #Model::Model/Upsample/1551
        self.module_203 = py_nndct.nn.Module('cast') #Model::Model/Upsample/1556
        self.module_204 = py_nndct.nn.Module('const') #Model::1557
        self.module_205 = py_nndct.nn.Module('mul') #Model::Model/Upsample/1558
        self.module_206 = py_nndct.nn.Module('cast') #Model::Model/Upsample/1563
        self.module_207 = py_nndct.nn.Module('floor') #Model::Model/Upsample/1564
        self.module_208 = py_nndct.nn.Int() #Model::Model/Upsample/1565
        self.module_209 = py_nndct.nn.Module('shape') #Model::Model/Upsample/1567
        self.module_210 = py_nndct.nn.Module('tensor') #Model::Model/Upsample/1568
        self.module_211 = py_nndct.nn.Module('cast') #Model::Model/Upsample/1573
        self.module_212 = py_nndct.nn.Module('const') #Model::1574
        self.module_213 = py_nndct.nn.Module('mul') #Model::Model/Upsample/1575
        self.module_214 = py_nndct.nn.Module('cast') #Model::Model/Upsample/1580
        self.module_215 = py_nndct.nn.Module('floor') #Model::Model/Upsample/1581
        self.module_216 = py_nndct.nn.Int() #Model::Model/Upsample/1582
        self.module_217 = py_nndct.nn.Interpolate() #Model::Model/Upsample/1584
        self.module_218 = py_nndct.nn.Cat() #Model::Model/Concat/input.178
        self.module_219 = py_nndct.nn.Conv2d(in_channels=768, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.179
        self.module_221 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.181
        self.module_222 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.182
        self.module_224 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/input.184
        self.module_225 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.185
        self.module_227 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.187
        self.module_228 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.188
        self.module_230 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/input.190
        self.module_231 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.191
        self.module_233 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.193
        self.module_234 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.194
        self.module_236 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.218
        self.module_237 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv/Conv2d[conv]/input.196
        self.module_239 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Conv/LeakyReLU[act]/input.198
        self.module_240 = py_nndct.nn.Module('shape') #Model::Model/Upsample/1715
        self.module_241 = py_nndct.nn.Module('tensor') #Model::Model/Upsample/1716
        self.module_242 = py_nndct.nn.Module('cast') #Model::Model/Upsample/1721
        self.module_243 = py_nndct.nn.Module('const') #Model::1722
        self.module_244 = py_nndct.nn.Module('mul') #Model::Model/Upsample/1723
        self.module_245 = py_nndct.nn.Module('cast') #Model::Model/Upsample/1728
        self.module_246 = py_nndct.nn.Module('floor') #Model::Model/Upsample/1729
        self.module_247 = py_nndct.nn.Int() #Model::Model/Upsample/1730
        self.module_248 = py_nndct.nn.Module('shape') #Model::Model/Upsample/1732
        self.module_249 = py_nndct.nn.Module('tensor') #Model::Model/Upsample/1733
        self.module_250 = py_nndct.nn.Module('cast') #Model::Model/Upsample/1738
        self.module_251 = py_nndct.nn.Module('const') #Model::1739
        self.module_252 = py_nndct.nn.Module('mul') #Model::Model/Upsample/1740
        self.module_253 = py_nndct.nn.Module('cast') #Model::Model/Upsample/1745
        self.module_254 = py_nndct.nn.Module('floor') #Model::Model/Upsample/1746
        self.module_255 = py_nndct.nn.Int() #Model::Model/Upsample/1747
        self.module_256 = py_nndct.nn.Interpolate() #Model::Model/Upsample/1749
        self.module_257 = py_nndct.nn.Cat() #Model::Model/Concat/input.199
        self.module_258 = py_nndct.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.200
        self.module_260 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.202
        self.module_261 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.203
        self.module_263 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/input.205
        self.module_264 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.206
        self.module_266 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.208
        self.module_267 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.209
        self.module_269 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.211
        self.module_270 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.212
        self.module_272 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.214
        self.module_273 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.215
        self.module_275 = py_nndct.nn.LeakyReLU(negative_slope=0.1, inplace=True) #Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/input.217
        self.module_276 = py_nndct.nn.Conv2d(in_channels=256, out_channels=18, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Detect/Conv2d/1870
        self.module_277 = py_nndct.nn.Module('shape') #Model::Model/Detect/1872
        self.module_278 = py_nndct.nn.Module('shape') #Model::Model/Detect/1874
        self.module_279 = py_nndct.nn.Module('shape') #Model::Model/Detect/1876
        self.module_280 = py_nndct.nn.Module('reshape') #Model::Model/Detect/1880
        self.module_281 = py_nndct.nn.Module('permute') #Model::Model/Detect/1882
        self.module_282 = py_nndct.nn.Module('contiguous') #Model::Model/Detect/1884
        self.module_283 = py_nndct.nn.Conv2d(in_channels=512, out_channels=18, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Detect/Conv2d/1894
        self.module_284 = py_nndct.nn.Module('shape') #Model::Model/Detect/1896
        self.module_285 = py_nndct.nn.Module('shape') #Model::Model/Detect/1898
        self.module_286 = py_nndct.nn.Module('shape') #Model::Model/Detect/1900
        self.module_287 = py_nndct.nn.Module('reshape') #Model::Model/Detect/1904
        self.module_288 = py_nndct.nn.Module('permute') #Model::Model/Detect/1906
        self.module_289 = py_nndct.nn.Module('contiguous') #Model::Model/Detect/1908
        self.module_290 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=18, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Detect/Conv2d/1918
        self.module_291 = py_nndct.nn.Module('shape') #Model::Model/Detect/1920
        self.module_292 = py_nndct.nn.Module('shape') #Model::Model/Detect/1922
        self.module_293 = py_nndct.nn.Module('shape') #Model::Model/Detect/1924
        self.module_294 = py_nndct.nn.Module('reshape') #Model::Model/Detect/1928
        self.module_295 = py_nndct.nn.Module('permute') #Model::Model/Detect/1930
        self.module_296 = py_nndct.nn.Module('contiguous') #Model::Model/Detect/1932

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_3 = self.module_3(self.output_module_1)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_6 = self.module_6(self.output_module_4)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_9 = self.module_9(self.output_module_7)
        self.output_module_10 = self.module_10(self.output_module_9)
        self.output_module_12 = self.module_12(self.output_module_10)
        self.output_module_13 = self.module_13(alpha=1, input=self.output_module_6, other=self.output_module_12)
        self.output_module_14 = self.module_14(self.output_module_13)
        self.output_module_16 = self.module_16(self.output_module_14)
        self.output_module_17 = self.module_17(self.output_module_16)
        self.output_module_19 = self.module_19(self.output_module_17)
        self.output_module_20 = self.module_20(self.output_module_19)
        self.output_module_22 = self.module_22(self.output_module_20)
        self.output_module_23 = self.module_23(alpha=1, input=self.output_module_16, other=self.output_module_22)
        self.output_module_24 = self.module_24(self.output_module_23)
        self.output_module_26 = self.module_26(self.output_module_24)
        self.output_module_27 = self.module_27(self.output_module_26)
        self.output_module_29 = self.module_29(self.output_module_27)
        self.output_module_30 = self.module_30(alpha=1, input=self.output_module_23, other=self.output_module_29)
        self.output_module_31 = self.module_31(self.output_module_30)
        self.output_module_33 = self.module_33(self.output_module_31)
        self.output_module_34 = self.module_34(self.output_module_33)
        self.output_module_36 = self.module_36(self.output_module_34)
        self.output_module_37 = self.module_37(self.output_module_36)
        self.output_module_39 = self.module_39(self.output_module_37)
        self.output_module_40 = self.module_40(alpha=1, input=self.output_module_33, other=self.output_module_39)
        self.output_module_41 = self.module_41(self.output_module_40)
        self.output_module_43 = self.module_43(self.output_module_41)
        self.output_module_44 = self.module_44(self.output_module_43)
        self.output_module_46 = self.module_46(self.output_module_44)
        self.output_module_47 = self.module_47(alpha=1, input=self.output_module_40, other=self.output_module_46)
        self.output_module_48 = self.module_48(self.output_module_47)
        self.output_module_50 = self.module_50(self.output_module_48)
        self.output_module_51 = self.module_51(self.output_module_50)
        self.output_module_53 = self.module_53(self.output_module_51)
        self.output_module_54 = self.module_54(alpha=1, input=self.output_module_47, other=self.output_module_53)
        self.output_module_55 = self.module_55(self.output_module_54)
        self.output_module_57 = self.module_57(self.output_module_55)
        self.output_module_58 = self.module_58(self.output_module_57)
        self.output_module_60 = self.module_60(self.output_module_58)
        self.output_module_61 = self.module_61(alpha=1, input=self.output_module_54, other=self.output_module_60)
        self.output_module_62 = self.module_62(self.output_module_61)
        self.output_module_64 = self.module_64(self.output_module_62)
        self.output_module_65 = self.module_65(self.output_module_64)
        self.output_module_67 = self.module_67(self.output_module_65)
        self.output_module_68 = self.module_68(alpha=1, input=self.output_module_61, other=self.output_module_67)
        self.output_module_69 = self.module_69(self.output_module_68)
        self.output_module_71 = self.module_71(self.output_module_69)
        self.output_module_72 = self.module_72(self.output_module_71)
        self.output_module_74 = self.module_74(self.output_module_72)
        self.output_module_75 = self.module_75(alpha=1, input=self.output_module_68, other=self.output_module_74)
        self.output_module_76 = self.module_76(self.output_module_75)
        self.output_module_78 = self.module_78(self.output_module_76)
        self.output_module_79 = self.module_79(self.output_module_78)
        self.output_module_81 = self.module_81(self.output_module_79)
        self.output_module_82 = self.module_82(alpha=1, input=self.output_module_75, other=self.output_module_81)
        self.output_module_83 = self.module_83(self.output_module_82)
        self.output_module_85 = self.module_85(self.output_module_83)
        self.output_module_86 = self.module_86(self.output_module_85)
        self.output_module_88 = self.module_88(self.output_module_86)
        self.output_module_89 = self.module_89(alpha=1, input=self.output_module_82, other=self.output_module_88)
        self.output_module_90 = self.module_90(self.output_module_89)
        self.output_module_92 = self.module_92(self.output_module_90)
        self.output_module_93 = self.module_93(self.output_module_92)
        self.output_module_95 = self.module_95(self.output_module_93)
        self.output_module_96 = self.module_96(self.output_module_95)
        self.output_module_98 = self.module_98(self.output_module_96)
        self.output_module_99 = self.module_99(alpha=1, input=self.output_module_92, other=self.output_module_98)
        self.output_module_100 = self.module_100(self.output_module_99)
        self.output_module_102 = self.module_102(self.output_module_100)
        self.output_module_103 = self.module_103(self.output_module_102)
        self.output_module_105 = self.module_105(self.output_module_103)
        self.output_module_106 = self.module_106(alpha=1, input=self.output_module_99, other=self.output_module_105)
        self.output_module_107 = self.module_107(self.output_module_106)
        self.output_module_109 = self.module_109(self.output_module_107)
        self.output_module_110 = self.module_110(self.output_module_109)
        self.output_module_112 = self.module_112(self.output_module_110)
        self.output_module_113 = self.module_113(alpha=1, input=self.output_module_106, other=self.output_module_112)
        self.output_module_114 = self.module_114(self.output_module_113)
        self.output_module_116 = self.module_116(self.output_module_114)
        self.output_module_117 = self.module_117(self.output_module_116)
        self.output_module_119 = self.module_119(self.output_module_117)
        self.output_module_120 = self.module_120(alpha=1, input=self.output_module_113, other=self.output_module_119)
        self.output_module_121 = self.module_121(self.output_module_120)
        self.output_module_123 = self.module_123(self.output_module_121)
        self.output_module_124 = self.module_124(self.output_module_123)
        self.output_module_126 = self.module_126(self.output_module_124)
        self.output_module_127 = self.module_127(alpha=1, input=self.output_module_120, other=self.output_module_126)
        self.output_module_128 = self.module_128(self.output_module_127)
        self.output_module_130 = self.module_130(self.output_module_128)
        self.output_module_131 = self.module_131(self.output_module_130)
        self.output_module_133 = self.module_133(self.output_module_131)
        self.output_module_134 = self.module_134(alpha=1, input=self.output_module_127, other=self.output_module_133)
        self.output_module_135 = self.module_135(self.output_module_134)
        self.output_module_137 = self.module_137(self.output_module_135)
        self.output_module_138 = self.module_138(self.output_module_137)
        self.output_module_140 = self.module_140(self.output_module_138)
        self.output_module_141 = self.module_141(alpha=1, input=self.output_module_134, other=self.output_module_140)
        self.output_module_142 = self.module_142(self.output_module_141)
        self.output_module_144 = self.module_144(self.output_module_142)
        self.output_module_145 = self.module_145(self.output_module_144)
        self.output_module_147 = self.module_147(self.output_module_145)
        self.output_module_148 = self.module_148(alpha=1, input=self.output_module_141, other=self.output_module_147)
        self.output_module_149 = self.module_149(self.output_module_148)
        self.output_module_151 = self.module_151(self.output_module_149)
        self.output_module_152 = self.module_152(self.output_module_151)
        self.output_module_154 = self.module_154(self.output_module_152)
        self.output_module_155 = self.module_155(self.output_module_154)
        self.output_module_157 = self.module_157(self.output_module_155)
        self.output_module_158 = self.module_158(alpha=1, input=self.output_module_151, other=self.output_module_157)
        self.output_module_159 = self.module_159(self.output_module_158)
        self.output_module_161 = self.module_161(self.output_module_159)
        self.output_module_162 = self.module_162(self.output_module_161)
        self.output_module_164 = self.module_164(self.output_module_162)
        self.output_module_165 = self.module_165(alpha=1, input=self.output_module_158, other=self.output_module_164)
        self.output_module_166 = self.module_166(self.output_module_165)
        self.output_module_168 = self.module_168(self.output_module_166)
        self.output_module_169 = self.module_169(self.output_module_168)
        self.output_module_171 = self.module_171(self.output_module_169)
        self.output_module_172 = self.module_172(alpha=1, input=self.output_module_165, other=self.output_module_171)
        self.output_module_173 = self.module_173(self.output_module_172)
        self.output_module_175 = self.module_175(self.output_module_173)
        self.output_module_176 = self.module_176(self.output_module_175)
        self.output_module_178 = self.module_178(self.output_module_176)
        self.output_module_179 = self.module_179(alpha=1, input=self.output_module_172, other=self.output_module_178)
        self.output_module_180 = self.module_180(self.output_module_179)
        self.output_module_182 = self.module_182(self.output_module_180)
        self.output_module_183 = self.module_183(self.output_module_182)
        self.output_module_185 = self.module_185(self.output_module_183)
        self.output_module_186 = self.module_186(self.output_module_185)
        self.output_module_188 = self.module_188(self.output_module_186)
        self.output_module_189 = self.module_189(self.output_module_188)
        self.output_module_191 = self.module_191(self.output_module_189)
        self.output_module_192 = self.module_192(self.output_module_191)
        self.output_module_194 = self.module_194(self.output_module_192)
        self.output_module_195 = self.module_195(self.output_module_194)
        self.output_module_197 = self.module_197(self.output_module_195)
        self.output_module_198 = self.module_198(self.output_module_194)
        self.output_module_200 = self.module_200(self.output_module_198)
        self.output_module_201 = self.module_201(input=self.output_module_200, dim=2)
        self.output_module_202 = self.module_202(dtype=torch.float32, device='cuda', data=self.output_module_201)
        self.output_module_203 = self.module_203(input=self.output_module_202, dtype=torch.float)
        self.output_module_204 = self.module_204(dtype=torch.float, device='cuda', data=2.0)
        self.output_module_205 = self.module_205(input=self.output_module_203, other=self.output_module_204)
        self.output_module_206 = self.module_206(input=self.output_module_205, dtype=torch.float)
        self.output_module_207 = self.module_207(input=self.output_module_206)
        self.output_module_208 = self.module_208(input=self.output_module_207)
        self.output_module_209 = self.module_209(input=self.output_module_200, dim=3)
        self.output_module_210 = self.module_210(dtype=torch.float32, device='cuda', data=self.output_module_209)
        self.output_module_211 = self.module_211(input=self.output_module_210, dtype=torch.float)
        self.output_module_212 = self.module_212(dtype=torch.float, device='cuda', data=2.0)
        self.output_module_213 = self.module_213(input=self.output_module_211, other=self.output_module_212)
        self.output_module_214 = self.module_214(input=self.output_module_213, dtype=torch.float)
        self.output_module_215 = self.module_215(input=self.output_module_214)
        self.output_module_216 = self.module_216(input=self.output_module_215)
        self.output_module_217 = self.module_217(input=self.output_module_200, size=[self.output_module_208,self.output_module_216], scale_factor=None, mode='nearest')
        self.output_module_218 = self.module_218(tensors=[self.output_module_217,self.output_module_148], dim=1)
        self.output_module_219 = self.module_219(self.output_module_218)
        self.output_module_221 = self.module_221(self.output_module_219)
        self.output_module_222 = self.module_222(self.output_module_221)
        self.output_module_224 = self.module_224(self.output_module_222)
        self.output_module_225 = self.module_225(self.output_module_224)
        self.output_module_227 = self.module_227(self.output_module_225)
        self.output_module_228 = self.module_228(self.output_module_227)
        self.output_module_230 = self.module_230(self.output_module_228)
        self.output_module_231 = self.module_231(self.output_module_230)
        self.output_module_233 = self.module_233(self.output_module_231)
        self.output_module_234 = self.module_234(self.output_module_233)
        self.output_module_236 = self.module_236(self.output_module_234)
        self.output_module_237 = self.module_237(self.output_module_233)
        self.output_module_239 = self.module_239(self.output_module_237)
        self.output_module_240 = self.module_240(input=self.output_module_239, dim=2)
        self.output_module_241 = self.module_241(dtype=torch.float32, device='cuda', data=self.output_module_240)
        self.output_module_242 = self.module_242(input=self.output_module_241, dtype=torch.float)
        self.output_module_243 = self.module_243(dtype=torch.float, device='cuda', data=2.0)
        self.output_module_244 = self.module_244(input=self.output_module_242, other=self.output_module_243)
        self.output_module_245 = self.module_245(input=self.output_module_244, dtype=torch.float)
        self.output_module_246 = self.module_246(input=self.output_module_245)
        self.output_module_247 = self.module_247(input=self.output_module_246)
        self.output_module_248 = self.module_248(input=self.output_module_239, dim=3)
        self.output_module_249 = self.module_249(dtype=torch.float32, device='cuda', data=self.output_module_248)
        self.output_module_250 = self.module_250(input=self.output_module_249, dtype=torch.float)
        self.output_module_251 = self.module_251(dtype=torch.float, device='cuda', data=2.0)
        self.output_module_252 = self.module_252(input=self.output_module_250, other=self.output_module_251)
        self.output_module_253 = self.module_253(input=self.output_module_252, dtype=torch.float)
        self.output_module_254 = self.module_254(input=self.output_module_253)
        self.output_module_255 = self.module_255(input=self.output_module_254)
        self.output_module_256 = self.module_256(input=self.output_module_239, size=[self.output_module_247,self.output_module_255], scale_factor=None, mode='nearest')
        self.output_module_257 = self.module_257(tensors=[self.output_module_256,self.output_module_89], dim=1)
        self.output_module_258 = self.module_258(self.output_module_257)
        self.output_module_260 = self.module_260(self.output_module_258)
        self.output_module_261 = self.module_261(self.output_module_260)
        self.output_module_263 = self.module_263(self.output_module_261)
        self.output_module_264 = self.module_264(self.output_module_263)
        self.output_module_266 = self.module_266(self.output_module_264)
        self.output_module_267 = self.module_267(self.output_module_266)
        self.output_module_269 = self.module_269(self.output_module_267)
        self.output_module_270 = self.module_270(self.output_module_269)
        self.output_module_272 = self.module_272(self.output_module_270)
        self.output_module_273 = self.module_273(self.output_module_272)
        self.output_module_275 = self.module_275(self.output_module_273)
        self.output_module_276 = self.module_276(self.output_module_275)
        self.output_module_277 = self.module_277(input=self.output_module_276, dim=0)
        self.output_module_278 = self.module_278(input=self.output_module_276, dim=2)
        self.output_module_279 = self.module_279(input=self.output_module_276, dim=3)
        self.output_module_280 = self.module_280(input=self.output_module_276, size=[self.output_module_277,3,6,self.output_module_278,self.output_module_279])
        self.output_module_281 = self.module_281(input=self.output_module_280, dims=[0,1,3,4,2])
        self.output_module_282 = self.module_282(input=self.output_module_281, )
        self.output_module_283 = self.module_283(self.output_module_236)
        self.output_module_284 = self.module_284(input=self.output_module_283, dim=0)
        self.output_module_285 = self.module_285(input=self.output_module_283, dim=2)
        self.output_module_286 = self.module_286(input=self.output_module_283, dim=3)
        self.output_module_287 = self.module_287(input=self.output_module_283, size=[self.output_module_284,3,6,self.output_module_285,self.output_module_286])
        self.output_module_288 = self.module_288(input=self.output_module_287, dims=[0,1,3,4,2])
        self.output_module_289 = self.module_289(input=self.output_module_288, )
        self.output_module_290 = self.module_290(self.output_module_197)
        self.output_module_291 = self.module_291(input=self.output_module_290, dim=0)
        self.output_module_292 = self.module_292(input=self.output_module_290, dim=2)
        self.output_module_293 = self.module_293(input=self.output_module_290, dim=3)
        self.output_module_294 = self.module_294(input=self.output_module_290, size=[self.output_module_291,3,6,self.output_module_292,self.output_module_293])
        self.output_module_295 = self.module_295(input=self.output_module_294, dims=[0,1,3,4,2])
        self.output_module_296 = self.module_296(input=self.output_module_295, )
        return self.output_module_282,self.output_module_289,self.output_module_296
