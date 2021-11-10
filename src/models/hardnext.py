# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.modules as F

def Round_out_ch(out_ch):
    result = int(int(out_ch+1)/2)*2
    return result

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)

class ConvLayer(nn.Sequential): # conv + batchnorm + relu6
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, bias=False):
        super().__init__()
        #out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size=kernel,          
                                          stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))  # True 表示 inplace 的意思                                       
    def forward(self, x):
        return super().forward(x)

class ConvLayer_no_activation(nn.Sequential): # conv + batchnorm + relu6
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, bias=False):
        super().__init__()
        #out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size=kernel,          
                                          stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        #self.add_module('relu', nn.ReLU6(True))  # True 表示 inplace 的意思                                       
    def forward(self, x):
        return super().forward(x)

class HarDBlock_3_layer(nn.Module):
    def __init__(self, in_ch, kernel=3, bias=False, k=32, m_1=4, downsample = 0):
        super().__init__()

        self.k_1 = k*m_1 #Round_out_ch(k*(m**2))
        self.downsample = downsample

        
        self.blk_out = k + self.k_1
        
        self.conv1= ConvLayer(in_ch=in_ch, out_ch=k, kernel=kernel)
        self.conv2= ConvLayer(in_ch=k, out_ch=k, kernel=kernel)
        self.conv3= ConvLayer(in_ch=(in_ch+k), out_ch=self.k_1, kernel=kernel)
    
    
    def forward(self, x): 
        identity = x

        out_1 = self.conv1(x)
        out = self.conv2(out_1)
        out = self.conv3(torch.cat([identity, out], 1))

       
        final_out = torch.cat([out_1, out], 1)
        

        return final_out

class HarDBlock_6_layer(nn.Module):
    def __init__(self, in_ch, kernel=3, bias=False, k=32, m_1=4, m_2=6, downsample=False):
        super().__init__()

        self.k_1 = k*m_1#Round_out_ch(k*m)
        self.k_2 = k*m_2#Round_out_ch(k*(m**2))
        
        self.blk_out = k*2 + self.k_2

        self.conv1 = ConvLayer(in_ch=in_ch, out_ch=k, kernel=kernel)
        self.conv2 = ConvLayer(in_ch=k, out_ch=k, kernel=kernel)
        self.conv3 = ConvLayer(in_ch=(in_ch + k), out_ch=self.k_1, kernel=kernel)

        self.conv4 = ConvLayer(in_ch=self.k_1, out_ch=k, kernel=kernel)
        self.conv5 = ConvLayer(in_ch=k, out_ch=k, kernel=kernel)
        self.conv6 = ConvLayer(in_ch=(in_ch + self.k_1 + k), out_ch=self.k_2, kernel=kernel)

    def forward(self, x):
        identity = x

        t_out_1 = self.conv1(x)
        out     = self.conv2(t_out_1)
        out_2   = self.conv3(torch.cat([identity, out], 1))
        t_out_2 = self.conv4(out_2)
        out     = self.conv5(t_out_2)
        out     = self.conv6(torch.cat([identity, out_2, out], 1))

        final_out = torch.cat([t_out_1, t_out_2, out], 1)

        return final_out

class HarDBlock_12_layer(nn.Module):
    def __init__(self, in_ch, kernel=3, bias=False, k=32, m_1=2, m_2=4, m_3=6, downsample=False):
        super().__init__()

        
        
        self.k_1 = k*m_1
        self.k_2 = k*m_2
        self.k_3 = k*m_3

        self.blk_out = k*4 + self.k_3

        self.conv1 = ConvLayer(in_ch=in_ch, out_ch=k, kernel=kernel)
        self.conv2 = ConvLayer(in_ch=k, out_ch=k, kernel=kernel)
        self.conv3 = ConvLayer(in_ch=(in_ch + k), out_ch=self.k_1, kernel=kernel)

        self.conv4 = ConvLayer(in_ch=self.k_1, out_ch=k, kernel=kernel)
        self.conv5 = ConvLayer(in_ch=k, out_ch=k, kernel=kernel)
        self.conv6 = ConvLayer(in_ch=(in_ch + self.k_1 + k), out_ch=self.k_2, kernel=kernel)

        self.conv7 = ConvLayer(in_ch=self.k_2, out_ch=k, kernel=kernel)
        self.conv8 = ConvLayer(in_ch=k, out_ch=k, kernel=kernel)
        self.conv9 = ConvLayer(in_ch=(self.k_2 + k), out_ch=self.k_1, kernel=kernel)

        self.conv10 = ConvLayer(in_ch=self.k_1, out_ch=k, kernel=kernel)
        self.conv11 = ConvLayer(in_ch=k, out_ch=k, kernel=kernel)
        self.conv12 = ConvLayer(in_ch=(in_ch + self.k_2 + self.k_1 + k), out_ch=self.k_3, kernel=kernel)

    def forward(self, x):

        identity = x

        t_out_1 = self.conv1(x)
        out     = self.conv2(t_out_1)
        out_2   = self.conv3(torch.cat([identity, out], 1))
        t_out_2 = self.conv4(out_2)
        out     = self.conv5(t_out_2)
        out_3   = self.conv6(torch.cat([identity, out_2, out], 1))

        t_out_3 = self.conv7(out_3)
        out     = self.conv8(t_out_3)
        out_4   = self.conv9(torch.cat([out_3, out], 1))
        t_out_4 = self.conv10(out_4)
        out     = self.conv11(t_out_4)
        out     = self.conv12(torch.cat([identity, out_3, out_4, out], 1))

        final_out = torch.cat([t_out_1, t_out_2, t_out_3, t_out_4 ,out], 1)

        return final_out

class HarDNeXt(nn.Module):
    def __init__(self, depth_wise = False, arch=28):
        super().__init__()

        # STEM block
        first_ch  = [32, 64]
        second_kernel = 3
        # max_pool = True
        grmul = 1.7
        drop_rate = 0.1


        if arch == 28:
            
            first_ch  = [24, 48]
            ch_list =  [  48, 144,  216,  648,    1280]
            gr       = [  16,  24,   72,  108,     162]
            m1       = [   2,   2,    2,    2,       4]
            m2       = [   0,   4,    0,    4,       0]
            n_layers = [   3,   6,    3,    6,       3]
            kernel   = [   3,   5,    3,    3,       3]
            downSamp = [   1,   0,    1,    1,       0]

        
        
        elif arch == 32:
            
            first_ch  = [24, 48]
            ch_list =  [  64, 112,  224, 256,  560,    1024]
            gr       = [  16,  16,   56,  64,   80,     160]
            m1       = [   3,   3,    3,   3,    3,       5]
            m2       = [   0,   5,    0,   0,    5,       0]
            n_layers = [   3,   6,    3,   3,    6,       3]
            kernel   = [   3,   5,    3,   3,    3,       3]
            downSamp = [   1,   0,    1,   0,    1,       0]
        
        
        elif arch == 39:
            ''' 75.37 %'''
            first_ch  = [24, 48]
            ch_list =  [  64, 168,  192,  448,  560,     1024]
            gr       = [  16,  24,   48,   64,   80,      160]
            m1       = [   3,   3,    3,    3,    3,        5]
            m2       = [   0,   5,    0,    5,    5,        0]
            n_layers = [   3,   6,    3,    6,    6,        3]
            kernel   = [   3,   5,    3,    3,    3,        3]
            downSamp = [   1,   0,    1,    0,    1,        0]
        
        elif arch == 50:
            ''' 502  76.32%, 192 FPS'''
            
            first_ch  = [32, 64]
            ch_list  = [  64, 128,  192, 288, 768,  1024]
            gr       = [  16,  16,   32,  48,  96,   192]
            m1       = [   3,   2,    3,   3,   2,     4]
            m2       = [   0,   3,    4,   4,   3,     0]
            m3       = [   0,   4,    0,   0,   4,     0]
            n_layers = [   3,  12,    6,   6,  12,     3]
            kernel   = [   3,   5,    3,   3,   3,     3]
            downSamp = [   1,   0,    1,   0,   1,     0]

        elif arch == 56:
            
            first_ch  = [48, 96]
            ch_list =  [ 120, 192,  288, 512, 768,  1024]
            gr       = [  24,  24,   48,  64,  96,   256]
            m1       = [   4,   2,    3,   2,   2,     3]
            m2       = [   0,   3,    4,   3,   3,     0]
            m3       = [   0,   4,    0,   4,   4,     0]
            n_layers = [   3,  12,    6,  12,  12,     3]
            kernel   = [   3,   5,    3,   3,   3,     3]
            downSamp = [   1,   0,    1,   0,   1,     0]
        
        
        
        
        
        

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        self.base.append(ConvLayer(in_ch=3, out_ch=first_ch[0], kernel=3, stride=2, bias=False))
        self.base.append(ConvLayer(in_ch=first_ch[0], out_ch=first_ch[1], kernel=second_kernel))
        self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ch = first_ch[1]
        

        for i in range(blks):

            if n_layers[i] == 3:
                blk = HarDBlock_3_layer(in_ch=ch, kernel=kernel[i], bias=False, k=gr[i], m_1=m1[i])
            
            elif n_layers[i] == 6:
                blk = HarDBlock_6_layer(in_ch=ch, kernel=kernel[i], bias=False, k=gr[i], m_1=m1[i], m_2=m2[i])
            
            elif n_layers[i] == 12:
                blk = HarDBlock_12_layer(in_ch=ch, kernel=kernel[i], bias=False, k=gr[i], m_1=m1[i], m_2=m2[i], m_3=m3[i])
            
            ch = blk.blk_out
            self.base.append(blk)
            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) ) # trainsition layer

            ch = ch_list[i]

            if downSamp[i] == 1:
                self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

        ch = ch_list[blks-1]
        self.base.append (
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 1000) ))
    
    def forward(self, x):
        # for layer in self.base:
        #   x = layer(x)
        for i in range(len(self.base)):
          x = self.base[i](x)
        return x
