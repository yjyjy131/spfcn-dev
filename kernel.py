import torch
from torch import nn


# nn.Module : custom 신경망 설계 및 구현
# module : 한 개 이상의 레이어로 구성됨
# 신경망은 한개 이상의 모듈로 이루어짐
class BasicModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, stride=1):
        nn.Module.__init__(self)
        # padding?
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=(kernel - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


# selectKernel 은 basic module 2개를 붙인 구조임
# 따라서 conv - bn - relu - conv - bn - relu 구조가 select kernel 하나를 이룸
# 논문 내 select module
class SelectKernel(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        nn.Module.__init__(self)
        self.conv3 = BasicModule(in_channel, out_channel, 3, stride)
        self.conv5 = BasicModule(in_channel, out_channel, 5, stride)
        # training 단계에서만 사용됨
        # TODO : Linear output 2?
        self.cen = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2),
            nn.Softmax(dim=0)
        )

    # 위 레이어들을 활용해 논문내 vector alpha 를 계산하는 것
    def forward(self, x):
        # CEN layer 들어가기 전 vector 값을 계산
        # TODO : vector 를 계산할 때는 feature x 값이 필요 없다 ?
        # view(1) nx1 차원 텐서를 1x1 텐서로 변환 [x]
        vector = torch.cat(
            [torch.mean(self.conv3.bn.running_mean).view(1), torch.std(self.conv3.bn.running_mean).view(1),
             torch.mean(self.conv3.bn.running_var).view(1), torch.std(self.conv3.bn.running_var).view(1),
             torch.mean(self.conv5.bn.running_mean).view(1), torch.std(self.conv5.bn.running_mean).view(1),
             torch.mean(self.conv5.bn.running_var).view(1), torch.std(self.conv5.bn.running_var).view(1)]
        )
        print(f'select kernel vector : {vector}')
        print(f'select kernel vector shape : {vector.shape}')
        alpha = self.cen(vector)
        return alpha[0] * self.conv3(x) + alpha[1] * self.conv5(x)


# TODO : in, out channel id 의 필요성 ?
class SpModule(nn.Module):
    def __init__(self, in_channels_id, out_channels_id, device,
                 in_channels, out_channels, stride=1):
        nn.Module.__init__(self)
        self.conv_module = SelectKernel(in_channels_id, out_channels_id, stride).to(device)

    def forward(self, feature):
        return self.conv_module(feature)


if __name__ == '__main__':
    pass
