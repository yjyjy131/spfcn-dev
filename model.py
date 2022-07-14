import torch
import torch.nn.functional as f
from torch import nn

from kernel import BasicModule, SelectKernel, SpModule


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
    def __init__(self, ipc, sqz):
        nn.Module.__init__(self)
        self.att = SEBlock(ipc, sqz)
        self.conv = nn.Conv2d(ipc, 1, kernel_size=1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        return self.activate(self.conv(self.att(x)))


class DirectionHeading(nn.Module):
    def __init__(self, ipc, sqz):
        self.att = SEBlock(ipc, sqz)
        self.conv = nn.Conv2d(ipc, 2, kernel_size=1)
        self.activate = nn.Tanh()

    def forward(self, x):
        return self.activate(self.conv(self.att(x)))


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

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
                module.bias.data.fill_(0)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    # SpModule 의 SelectKernel 로 특정 모듈을 select
    # SelectKernel 의 auto_select() 를 사용 시 alpha 값이 큰 conv 레이어 반환
    # TODO : 성능이 좋다 = 알파가 크다 ?
    def auto_select(self):
        count = 0
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, SelectKernel):
                select_check = module.conv_module.auto_select()
                if select_check is not None:
                    module.conv_module = select_check
                    count += 1
        print("\tAuto Selected %d module(s)" % count)

    # TODO : auto_select 와 enforce_select 의 차이 ?
    def enforce_select(self):
        count = 0
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, SelectKernel):
                module.conv_module = module.conv_module.enforce_select()
                count += 1
        print("\tEnforce Selected %d module(s)" % count)

    # TODO : prune_indices 란 ?
    def prune(self, group_id, prune_indices):
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                module.prune(group_id, prune_indices)

    # TODO : prune_info.indices ? : <function Tensor.indices>
    # weight 기여도 alpha 를 기준, 기여도 낮으면 kernel 제거
    def prune_channel(self):
        prune_dict = dict()
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                if module.out_channels_id in prune_dict.keys():
                    prune_dict[module.out_channels_id] += module.conv_module.get_alpha()
                else:
                    prune_dict[module.out_channels_id] = module.conv_module.get_alpha()

        # prune_list : prune_dict 값에서 생성된 (key, val) 형태의 generator 
        # prune_list 의 prune_info 는 prune_dict 의 val 값을 의미
        # prune_info 값은 torch tensor 타입 이므로, indices 변수를 가진다
        prune_list = ((key, torch.min(value, dim=0)) for key, value in prune_dict.items() if key >= 0)
        min_group, min_value, min_index = 0, 1, 0
        for group_id, prune_info in prune_list:
            if prune_info.values < 0.02:
                print("\tAuto Pruned: Group {}, Channel {}, Contribution {:.3f}".format(group_id, prune_info.indices.item(), prune_info.values.item()))
                for module in self.modules():
                    if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                        module.prune(group_id, prune_info.indices)
                    elif prune_info.values < min_value:
                        min_group, min_value, min_index = group_id, prune_info.values, prune_info.indices
        pass

    def merge(self):
        for module in self.modules():
            if isinstance(module, SpModule) and isinstance(module.conv_module, BasicModule):
                module.conv_module.merge(module.device)

        # initialize_weights()
        # train()

        # 1. update_cost
        # 2. auto select : auto_trainer.update_cost(neg=0.1)
        # 3. enforce select
        # 4. prune
        # 5. fine tuning : auto_trainer.with_regularizatio
        # 6. merge

if __name__ == '__main__':
    pass