import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.random import normal
from math import sqrt
import argparse

channel_dim = 3
ndf = 64

class GlobalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvBlock, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2

        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        #combine two paths
        x = x_l + x_r
        return x

class ResidualBlock(nn.Module):
    def __init__(self, indim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(indim, indim*2, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(indim*2)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(indim*2, indim*2, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(indim*2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(indim*2, indim, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(indim)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        #parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        out = x + residual
        return out

class ResidualBlock_D(nn.Module):
    def __init__(self, indim):
        super(ResidualBlock_D, self).__init__()
        self.conv1 = nn.Conv2d(indim, indim*2, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(indim*2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(indim*2, indim*2, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(indim*2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(indim*2, indim, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(indim)
        self.relu3 = nn.ReLU(inplace=True)
        #parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        out = x + residual
        return out


class NetS(nn.Module):
    def __init__(self, ngpu):
        super(NetS, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv2d(channel_dim, ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
        )
        self.convblock1_1 = ResidualBlock(ndf)
        self.convblock2 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
        )
        self.convblock2_1 = ResidualBlock(ndf*2)
        self.convblock3 = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
        )
        self.convblock3_1 = ResidualBlock(ndf*4)
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
        )
        self.convblock4_1 = ResidualBlock(ndf*8)
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.convblock5_1 = ResidualBlock(ndf*8)
        self.convblock6 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.convblock6_1 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.convblock7 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 1 x 1
        )
        # self.convblock7_1 = ResidualBlock(ndf*32)
        self.convblock8 = nn.Sequential(
            # state size. (ndf*32) x 1 x 1
            nn.Conv2d(ndf * 32, ndf * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1 x 1
        )


        self.deconvblock1 = nn.Sequential(
            # state size. (ngf*8) x 1 x 1
            nn.ConvTranspose2d(ndf * 8, ndf * 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 1 x 1
        )
        self.deconvblock2 = nn.Sequential(
            # state size. (cat: ngf*32) x 1 x 1
            nn.Conv2d(ndf * 64 , ndf * 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 2 x 2
        )
        self.deconvblock2_1 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.ReLU(inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.deconvblock3 = nn.Sequential(
            # state size. (cat: ngf*16) x 2 x 2
            nn.Conv2d(ndf * 16 * 2, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
        )
        self.deconvblock3_1 = ResidualBlock_D(ndf*8)
        self.deconvblock4 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            GlobalConvBlock(ndf*8*2, ndf*8, (7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
        )
        self.deconvblock4_1 = ResidualBlock_D(ndf*8)
        self.deconvblock5 = nn.Sequential(
            # state size. (ngf*8) x 8 x 8
            GlobalConvBlock(ndf*8*2, ndf*4, (7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
        )
        self.deconvblock5_1 = ResidualBlock_D(ndf*4)
        self.deconvblock6 = nn.Sequential(
            # state size. (ngf*4) x 16 x 16
            GlobalConvBlock(ndf*4*2, ndf*2, (9, 9)),
            # nn.ConvTranspose2d(ndf * 4 * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
        )
        self.deconvblock6_1 = ResidualBlock_D(ndf*2)
        self.deconvblock7 = nn.Sequential(
            # state size. (ngf*2) x 32 x 32
            GlobalConvBlock(ndf*2*2, ndf, (9, 9)),
            # nn.ConvTranspose2d(ndf * 2 * 2,     ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
        )
        self.deconvblock7_1 = ResidualBlock_D(ndf)
        self.deconvblock8 = nn.Sequential(
            # state size. (ngf) x 64 x 64
            GlobalConvBlock(ndf*2, ndf, (11, 11)),
            # nn.ConvTranspose2d( ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
        )
        self.deconvblock8_1 = ResidualBlock_D(ndf)
        self.deconvblock9 = nn.Sequential(
            # state size. (ngf) x 128 x 128
            nn.Conv2d( ndf, 1, 5, 1, 2, bias=False),
            # state size. (channel_dim) x 128 x 128
            # nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, input):
        # for now it only supports one GPU
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
            encoder1 = self.convblock1(input)
            encoder1 = self.convblock1_1(encoder1)
            encoder2 = self.convblock2(encoder1)
            encoder2 = self.convblock2_1(encoder2)
            encoder3 = self.convblock3(encoder2)
            encoder3 = self.convblock3_1(encoder3)
            encoder4 = self.convblock4(encoder3)
            encoder4 = self.convblock4_1(encoder4)
            encoder5 = self.convblock5(encoder4)
            encoder5 = self.convblock5_1(encoder5)
            encoder6 = self.convblock6(encoder5)
            encoder6 = self.convblock6_1(encoder6) + encoder6
            encoder7 = self.convblock7(encoder6)
            encoder8 = self.convblock8(encoder7)

            decoder1 = self.deconvblock1(encoder8)
            decoder1 = torch.cat([encoder7,decoder1],1)
            decoder1 = F.upsample(decoder1, size = encoder6.size()[2:], mode='bilinear')
            decoder2 = self.deconvblock2(decoder1)
            decoder2 = self.deconvblock2_1(decoder2) + decoder2
            # concatenate along depth dimension
            decoder2 = torch.cat([encoder6,decoder2],1)
            decoder2 = F.upsample(decoder2, size = encoder5.size()[2:], mode='bilinear')
            decoder3 = self.deconvblock3(decoder2)
            decoder3 = self.deconvblock3_1(decoder3)
            decoder3 = torch.cat([encoder5,decoder3],1)
            decoder3 = F.upsample(decoder3, size = encoder4.size()[2:], mode='bilinear')
            decoder4 = self.deconvblock4(decoder3)
            decoder4 = self.deconvblock4_1(decoder4)
            decoder4 = torch.cat([encoder4,decoder4],1)
            decoder4 = F.upsample(decoder4, size = encoder3.size()[2:], mode='bilinear')
            decoder5 = self.deconvblock5(decoder4)
            decoder5 = self.deconvblock5_1(decoder5)
            decoder5 = torch.cat([encoder3,decoder5],1)
            decoder5 = F.upsample(decoder5, size = encoder2.size()[2:], mode='bilinear')
            decoder6 = self.deconvblock6(decoder5)
            decoder6 = self.deconvblock6_1(decoder6)
            decoder6 = torch.cat([encoder2,decoder6],1)
            decoder6 = F.upsample(decoder6, size = encoder1.size()[2:], mode='bilinear')
            decoder7 = self.deconvblock7(decoder6)
            decoder7 = self.deconvblock7_1(decoder7)
            decoder7 = torch.cat([encoder1,decoder7],1)
            decoder7 = F.upsample(decoder7, size = input.size()[2:], mode='bilinear')
            decoder8 = self.deconvblock8(decoder7)
            decoder8 = self.deconvblock8_1(decoder8)
            decoder9 = self.deconvblock9(decoder8)
        else:
            print('For now we only support one GPU')
        
        return decoder9



class NetC(nn.Module):
    def __init__(self, ngpu):
        super(NetC, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv2d(channel_dim, ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf) x 64 x 64
        )
        self.convblock1_1 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            GlobalConvBlock(ndf, ndf * 2, (13, 13)),
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 64 x 64
        )
        self.convblock2 = nn.Sequential(
            # state size. (ndf * 2) x 64 x 64
            nn.Conv2d(ndf * 1, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 32 x 32
        )
        self.convblock2_1 = nn.Sequential(
            # input is (ndf*2) x 32 x 32
            GlobalConvBlock(ndf * 2, ndf * 4, (11, 11)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 32 x 32
        )
        self.convblock3 = nn.Sequential(
            # state size. (ndf * 4) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 16 x 16
        )
        self.convblock3_1 = nn.Sequential(
            # input is (ndf*4) x 16 x 16
            GlobalConvBlock(ndf * 4, ndf * 8, (9, 9)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf * 8) x 16 x 16
        )
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*8) x 8 x 8
        )
        self.convblock4_1 = nn.Sequential(
            # input is (ndf*8) x 8 x 8
            GlobalConvBlock(ndf * 8, ndf * 16, (7, 7)),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 8 x 8
        )
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 4 x 4
        )
        self.convblock5_1 = nn.Sequential(
            # input is (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 4 x 4
        )
        self.convblock6 = nn.Sequential(
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 2 x 2
        )
        # self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
            batchsize = input.size()[0]
            out1 = self.convblock1(input)
            # out1 = self.convblock1_1(out1)
            out2 = self.convblock2(out1)
            # out2 = self.convblock2_1(out2)
            out3 = self.convblock3(out2)
            # out3 = self.convblock3_1(out3)
            out4 = self.convblock4(out3)
            # out4 = self.convblock4_1(out4)
            out5 = self.convblock5(out4)
            # out5 = self.convblock5_1(out5)
            out6 = self.convblock6(out5)
            # out6 = self.convblock6_1(out6) + out6
            output = torch.cat((input.view(batchsize,-1),1*out1.view(batchsize,-1),
                                2*out2.view(batchsize,-1),2*out3.view(batchsize,-1),
                                2*out4.view(batchsize,-1),2*out5.view(batchsize,-1),
                                4*out6.view(batchsize,-1)),1)
        else:
            print('For now we only support one GPU')

        return output

