import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
from torchsnooper import snoop
from .deeplab_resnet import resnet50_locate
from .vgg import vgg16_locate
from .aspp import ASPP

config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False],[True, True, True, True, True]], 'score': 128}

class GradualBottleNeck(nn.Module):
    def __init__(self,inchannel,outchannel,lines,origin_kernel=3,origin_stride=1,origin_padding=1):
        super(GradualBottleNeck,self).__init__()
        modules=[]
        origin=inchannel
        lines=[inchannel]+lines
        lines.append(outchannel)
        for i in range(len(lines)-1):
            cur_kernel=1 if i>0 else origin_kernel
            cur_stride=1 if i>0 else origin_stride
            cur_padding=0 if i>0 else origin_padding
            modules.append(nn.Sequential(nn.Conv2d(lines[i],lines[i+1],cur_kernel,cur_stride,cur_padding,bias=False)))
        self.part=nn.Sequential(*modules)
    def forward(self,x):
        x=self.part(x)
        return x

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
#             up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.GELU()))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):#list_x and resl are both list, list_x[0] run by module[0], and so on.
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, need_x2, need_fuse,need_bn):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        self.need_bn=need_bn
        self.mid_channels=[64]
        pools, convs, bns = [],[],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            #nn.Conv2d(k, self.mid_channels, 3, 1, 1, bias=False), nn.Conv2d(self.mid_channels, k, 1, 1, 1, bias=False)
            convs.append(GradualBottleNeck(k,k,self.mid_channels))
            bns.append(nn.BatchNorm2d(k))

        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.bns=nn.ModuleList(bns)
#         self.relu = nn.ReLU()
        self.relu=nn.GELU()
        #add bottleneck
        #nn.Sequential(nn.Conv2d(k, self.mid_channels, 3, 1, 1, bias=False), nn.Conv2d(self.mid_channels, k_out, 1))
        self.conv_sum = GradualBottleNeck(k,k_out,self.mid_channels)
        if self.need_fuse:
            #add bottleneck
            #nn.Sequential(nn.Conv2d(k_out, self.mid_channels, 3, 1, 1, bias=False),
            #              nn.Conv2d(self.mid_channels, k_out, 1))
            self.conv_sum_c = GradualBottleNeck(k_out,k_out,self.mid_channels)
    # @snoop()
    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))

            # 此处引入batch_norm
            if self.need_bn:
                y=self.bns[i](y)

    
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)#change channels
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, deep_pool_layers, score_layers = [], [], []
    convert_layers = ConvertLayer(config['convert'])

    for i in range(len(config['deep_pool'][0])):
        #append all element in the right list
        deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i],config['deep_pool'][4][i])]

    score_layers = ScoreLayer(config['score'])

    return vgg, convert_layers, deep_pool_layers, score_layers


class PoolNet(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, deep_pool_layers, score_layers):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers

        self.aspp=ASPP(512,512)

        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers
    # @snoop()
    def forward(self, x):
        x_size = x.size()
        conv2merge, infos = self.base(x)
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)

        aspp_output=conv2merge[4]

        conv2merge = conv2merge[::-1]

        edge_merge = []
        # merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        merge = self.deep_pool[0](aspp_output, conv2merge[1], infos[0])
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])

        merge = self.deep_pool[-1](merge)
        merge = self.score(merge, x_size)
        return merge

def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'resnet':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

# if __name__=='__main__':
#     batch_size=16
#     channel=3
#     tmp_tensor=torch.tensor(np.ones(batch_size,channel,300,400))
#     a=build_model('resnet')
#     print(a)