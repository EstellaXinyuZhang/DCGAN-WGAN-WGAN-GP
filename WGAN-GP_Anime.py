from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import autograd

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='data/faces', help='folder to train data')
parser.add_argument('--outf', default='imgs/WGAN-GP', help='folder to output images and model checkpoints')
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像读入与预处理
dataset = dset.ImageFolder(root=opt.data_path,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),  # 缩放图片，保持长宽比不变，最短边为64
                               transforms.CenterCrop(opt.imageSize),  # 中心裁剪，剪出64*64的图
                               transforms.ToTensor(),  # 转换为Tensor
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化到[-1,1]
                           ]))

dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=opt.batchSize,
                                         shuffle=True,
                                         drop_last=True)


class Generator(nn.Module):
    def __init__(self, ngf, nz):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # 定义NetG的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


netG = Generator(opt.ngf, opt.nz).to(device)


class Critic(nn.Module):
    def __init__(self, ndf):
        super(Critic, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


netD = Critic(opt.ndf).to(device)

n_critic = 5
b1 = 0.5
b2 = 0.999

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(b1, b2))
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(b1, b2))

one = torch.FloatTensor([1])
# print(one), print(one.shape)
minus_one = -1*one
# print(minus_one), print(minus_one.shape)


def calculate_gradient_penalty(netD, real_images, fake_images):
    lambda_term = 10
    eta = torch.FloatTensor(opt.batchSize, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(opt.batchSize, real_images.size(1), real_images.size(2), real_images.size(3))

    eta = eta.to(device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    interpolated = interpolated.to(device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty


for epoch in range(1, opt.epoch + 1):
    for i, (imgs, _) in enumerate(dataloader):
        for p in netD.parameters():
            p.requires_grad = True

        errD_fake = 0
        errD_real = 0
        Wasserstein_D = 0
        # 固定生成器G，训练鉴别器D
        for d_iter in range(n_critic):
            optimizerD.zero_grad()
            # for p in netD.parameters():
            #     p.data.clamp_(-clip_value, clip_value)
            # 让D尽可能的把真图片判别为1
            imgs = imgs.to(device)
            errD_real = netD(imgs)
            errD_real = errD_real.mean().view(1)
            errD_real.backward(minus_one.to(device)) # backward()函数中没有填入任何tensor值, 就相当于 backward(torch.tensor([1])) ），则 x.grad 就是 ∂obj∂x∣∣x=1∂obj∂x|x=1

            noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
            noise = noise.to(device)
            fake = netG(noise)  # 生成假图
            errD_fake = netD(fake)
            errD_fake = errD_fake.mean().view(1)
            errD_fake.backward(one.to(device))

            # Train with gradient penalty
            gradient_penalty = calculate_gradient_penalty(netD, imgs.data, fake.data)
            gradient_penalty.backward(retain_graph=True)

            errD = errD_fake - errD_real + gradient_penalty
            Wasserstein_D = errD_real - errD_fake
            optimizerD.step()

        # 固定鉴别器D，训练生成器G
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation

        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
        noise = noise.to(device)
        fake = netG(noise)  # 生成假图
        errG = netD(fake)
        errG = errG.mean().view(1)
        errG.backward(minus_one.to(device))
        errG = -errG
        optimizerG.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Wasserstein_D: %.3f'
                  % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item(), Wasserstein_D.item()))

    vutils.save_image(fake.data,
                  '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                  normalize=True)
    torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))
