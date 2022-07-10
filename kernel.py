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

        # cen 에서 최적의 dilation value 조합을 찾을 수 있도록 kernel 기여도 평가
        self.cen = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2),
            nn.Softmax(dim=0)
        )

        self.alpha = torch.Tensor([1., 0.])

    # vector alpha 를 계산
    def forward(self, x):
        vector = torch.cat(
            [torch.mean(self.conv3.bn.running_mean).view(1), torch.std(self.conv3.bn.running_mean).view(1),
             torch.mean(self.conv3.bn.running_var).view(1), torch.std(self.conv3.bn.running_var).view(1),
             torch.mean(self.conv5.bn.running_mean).view(1), torch.std(self.conv5.bn.running_mean).view(1),
             torch.mean(self.conv5.bn.running_var).view(1), torch.std(self.conv5.bn.running_var).view(1)]
        )
        print(f'select kernel vector : {vector}')
        print(f'select kernel vector shape : {vector.shape}')
        self.alpha = self.cen(vector)
        return self.alpha[0] * self.conv3(x) + self.alpha[1] * self.conv5(x)

    # dilation value 들의 효과적 조합을 찾기
    # to select the convolution kernel with the best receptive fields in a convolutional layer
    # has several candidate convolution kernels with different dilation values
    # and its task is to select the best one among them
    # alpha[0] ( conv3 ) vs alpha[1] ( conv5 )
    # alpha 가 큰 conv 를 return
    def auto_select(self, threshold=0.9):
        return self.conv3 if self.alpha[0] > threshold else self.conv5 if self.alpha[1] > threshold else None

    def enforce_select(self):
        return self.conv3 if self.alpha[0] > self.alpha[1] else self.conv5


# TODO : in, out channel id 의 필요성 ?
# TODO : SpModule 의 초기 conv_module 은 SelectKernel 인데 BasicModule 로 바뀌는 경우는 어떤 케이스?
class SpModule(nn.Module):
    def __init__(self, in_channels_id, out_channels_id, device,
                 in_channels, out_channels, stride=1):
        nn.Module.__init__(self)
        self.device = device
        self.conv_module = SelectKernel(in_channels_id, out_channels_id, stride).to(device)

    def forward(self, feature):
        return self.conv_module(feature)

    # SpModule - SelectKernel - BasicModule 의 prune_in_channels 로 갈 수 있나 ? ( 불가능 )
    # TODO : self.conv_module 이 BasicModule 인 경우만 가능 : channel_id 와 group_id 가 같은 경우 무조건 BasicModule 일까?
    def prune(self, group_id, prune_idx):
        if self.in_channels_id == group_id:
            self.conv_module.prune_in_channels(prune_idx, self.device)
        elif self.out_channels_id == group_id:
            self.conv_module.prune_out_channels(prune_idx, self.device)


if __name__ == '__main__':
    pass
