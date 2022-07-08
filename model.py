import torch
import torch.nn.functional as f
from torch import nn

from .kernel import BasicModule, SelectKernel, SpModule


# TODO : group id 가 -1, +1, 2, +1, -1 로 대칭 id ?
class Hourglass(nn.Module):
    def __init__(self, encoder_dim, group_id, device):
        curr_dim = encoder_dim[0]
        next_dim = encoder_dim[1]
        self.front = SpModule(group_id-1, group_id, curr_dim, next_dim, device, stride=2)
        self.middle = nn.Sequential(
            SpModule(group_id, group_id+1, next_dim, next_dim, device, stride=1),
            SpModule(group_id+1, group_id+2, next_dim, next_dim, device, stride=1),
            SpModule(group_id+2, group_id, next_dim, next_dim, device, stride=1)
        ) if len(encoder_dim) <= 2 else Hourglass(encoder_dim[1:], group_id+1, device)
        self.rear = SpModule(group_id, group_id-1, next_dim, curr_dim, device, stride=1)

# TODO : rear 계산 시 interpolate / return 값은 feature + rear ?
    def forward(self, x):
        front = self.front(x)
        middle = self.middle(front)
        rear = f.interpolate(self.rear(middle), scale_factor=2)
        return x + rear


# SENet 에서 사용된 block 을 차용함
class SEBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        nn.Module.__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(nn.Linear(in_feature, out_feature),
                                 nn.RELU(inplace=True),
                                 nn.Linear(out_feature, in_feature),
                                 nn.Sigmoid())

    def forward(self, x):
        attention = self.pooling(x)
        attention = self.att(attention)
        return attention * x


# channel_encoder[0], 16
class MarkHeading(nn.Module):
    def __init__(self, encoder_dim, num):
        nn.Module.__init__(self)
        self.attr = SEBlock(encoder_dim, num)
        self.conv = nn.Conv2d(encoder_dim, 1, kernel_size=1)
        pass

    def forward(self):
        pass


class DirectionHeading(nn.Module):
    def __init__(self):
        pass

    def forward(self, feature):
        pass


class SlotNetwork(nn.Module):
    def __init__(self, encoder, device_id=0):
        device = torch.device('cpu' if device_id < 0 else 'cuda:%d' % device_id)

        self.norm = nn.LayerNorm([3, 224, 224])

        self.conv1 = SpModule(-1, 0, device, 3, encoder[0])
        self.backbone = Hourglass(encoder, 1, 0)
        self.conv2 = SpModule(0, -1, encoder[0], encoder[0])

        self.mark_heading = MarkHeading()
        self.direction_heading = DirectionHeading()

    def forward(self, x):
        x = self.conv1(x)
        x = self.backbone(x)
        x = self.conv2(x)
        mark = self.mark_heading(x)
        direction = self.direction_heading(x)
        return mark, direction


if __name__ == '__main__':
    pass