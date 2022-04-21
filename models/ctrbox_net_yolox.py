import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet
from ipdb import set_trace
from yolox.yolo_pafpn import YOLOPAFPN

class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(CTRBOX, self).__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        # channels = [3, 64, 128, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.width = 1
        # self.base_network = resnet.resnet101(pretrained=pretrained)
        self.base_network = YOLOPAFPN(depth=1, width=self.width)
        self.dec_c1 = CombinationModule(256*self.width, 128*self.width, batch_norm=True)
        self.dec_c2 = CombinationModule(512*self.width, 256*self.width, batch_norm=True)
        self.dec_c3 = CombinationModule(1024*self.width, 512*self.width, batch_norm=True)
        # self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        self.conv1 = nn.Conv2d(128*self.width, 256*self.width, kernel_size=3, stride=1, padding=1, groups=128*self.width, bias=False)
        self.bn1 = nn.BatchNorm2d(128*self.width)
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # print(x[-1].shape)
        # print(x[-2].shape)
        # print(x[-3].shape)
        # print(x[-4].shape)

        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)

        # c4_combine = self.dec_c4(x[-1], x[-2])
        # c3_combine = self.dec_c3(c4_combine, x[-3])
        c3_combine = self.dec_c3(x[-1], x[-2])
        c2_combine = self.dec_c2(c3_combine, x[-3])
        c1_combine = self.dec_c1(c2_combine, x[-4])

        c1_combine_con = self.conv1(c1_combine)
        out = c1_combine_con
        # c1_combine_con = self.bn1(c1_combine_con)

        # out = torch.cat((c1_combine, c1_combine_con), 1)

        # print(c1_combine.shape)
        # print(c1_combine_con.shape)
        # print(out.shape)



        # print(c1_combine.shape) # torch.Size([8, 128, 152, 152])
        # print(c3_combine.shape, x[-3].shape, c2_combine.shape, x[-4].shape) # torch.Size([8, 512, 38, 38]) torch.Size([8, 256, 76, 76]) torch.Size([8, 256, 76, 76]) torch.Size([8, 128, 152, 152])
        # c2_combine = self.up(c2_combine)
        # set_trace()
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(out) # torch.Size([8, 256, 152, 152])
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
