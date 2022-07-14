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

    def rebuild_conv(self, weight, bias, device):
        out_channels, in_channels, _, _ = weight.shape
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              self.conv.kernel_size,
                              self.conv.stride,
                              self.conv.padding,
                              bias=True).to(device)
        self.conv.weight = nn.Parameter(weight)
        self.conv.bias = nn.Parameter(bias)

    # L1 norm 으로 weight 기여도 측정
    def get_alpha(self):
        return nn.Softmax(dim=0)(torch.cat([torch.norm(weight, p=2).view(1) for weight in self.conv.weight], dim=0))

    # index 넘버 컬럼 제거
    def prune_in_channels(self, index, device):
        w = torch.cat((self.conv.weight[:, 0:index], self.conv.weight[:, index+1:]), dim=1)
        b = self.conv.bias
        self.rebuild_conv(w, b, device)

    def prune_out_channels(self, index, device):
        w = torch.cat((self.conv.weight[0:index], self.conv.weight[index+1:]), dim=0)
        b = torch.cat((self.conv.bias[0:index], self.conv.bias[index+1]), dim=0)
        self.rebuild_conv(w, b, device)

        w = torch.cat((self.conv.weight[0:index], self.bn.weight[index+1:]), dim=0)
        b = torch.cat((self.conv.weight[0:index], self.bn.weight[index+1:]), dim=0)

        self.bn = nn.BatchNorm2d(self.bn.num_features-1).to(device)
        self.bn.weight = nn.Parameter(w)
        self.bn.bias = nn.Parameter(b)

    def merge(self, device):
        if not hasattr(self, 'bn'):
            return
        mean = self.bn.running_mean
        var_sqrt = torch.sqrt(self.bn.running_var + self.bn.eps)
        beta = self.bn.weight
        gamma = self.bn.bias
        del self.bn

        w = self.conv.weight * (beta / var_sqrt).reshape([self.conv.out_channels, 1, 1, 1])
        b = (self.conv.bias - mean) / var_sqrt * beta + gamma
        self.rebuild_conv(w, b, device)

        self.forward = lambda feature: self.activate(self.conv(feature))



# conv - bn - relu - conv - bn - relu 구조가 select kernel 하나를 이룸
# 논문 내 select module
class SelectKernel(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        nn.Module.__init__(self)
        self.conv3 = BasicModule(in_channel, out_channel, 3, stride)
        self.conv5 = BasicModule(in_channel, out_channel, 5, stride)
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
        self.alpha = self.cen(vector)
        return self.alpha[0] * self.conv3(x) + self.alpha[1] * self.conv5(x)

    # dilation value 들의 효과적 조합을 찾기
    # to select the convolution kernel with the best receptive fields in a convolutional layer
    # has several candidate convolution kernels with different dilation values
    # and its task is to select the best one among them
    # alpha[0] ( conv3 ) vs alpha[1] ( conv5 )
    # alpha 가 큰 conv 를 return

    # select k는 basic module 2개로 이루어짐(conv3, conv5)
    # select k의 역할은 enforce, auto
    # select 로 최적의 receptive 를 뽑아내는(알파가 큰 ) kernel = basic module 를 리턴함

    def auto_select(self, threshold=0.9):
        return self.conv3 if self.alpha[0] > threshold else self.conv5 if self.alpha[1] > threshold else None

    def enforce_select(self):
        return self.conv3 if self.alpha[0] > self.alpha[1] else self.conv5


# TODO : SpModule 의 초기 conv_module 은 SelectKernel 인데 BasicModule 로 바뀌는 경우는 어떤 케이스?
class SpModule(nn.Module):
    def __init__(self, in_channels_id, out_channels_id, device,
                 in_channels, out_channels, stride=1):
        nn.Module.__init__(self)
        self.in_channels_id = in_channels_id
        self.out_channels_id = out_channels_id
        self.device = device
        self.conv_module = SelectKernel(in_channels, out_channels, stride).to(device)

    def forward(self, feature):
        return self.conv_module(feature)

    # SpModule - SelectKernel - BasicModule 의 prune_in_channels 로 갈 수 있나 ? ( 불가능 )
    # TODO : self.conv_module 이 BasicModule 인 경우만 가능 : channel_id 와 group_id 가 같은 경우 무조건 BasicModule 일까?
    def prune(self, group_id, prune_idx):
        if self.in_channels_id == group_id:
            self.conv_module.prune_in_channels(prune_idx, self.device)
        elif self.out_channels_id == group_id:
            self.conv_module.prune_out_channels(prune_idx, self.device)

    def get_regularization(self):
        if isinstance(self.conv_module, SelectKernel):
            return -10 * torch.log(self.conv_module.alpha[0] ** 2 + self.conv_module.alpha[1] ** 2)
        elif isinstance(self.conv_module, BasicModule):
            return 0.01 * torch.norm(self.conv_module.conv.weight, p=1)



if __name__ == '__main__':
    pass
