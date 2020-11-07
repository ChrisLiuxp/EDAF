import torch
import torch.nn as nn
import math

class FCOSClsCenterHead(nn.Module):
    def __init__(self, inplanes, num_classes, num_layers=4, prior=0.01):
        super(FCOSClsCenterHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
            layers.append(nn.GroupNorm(32, inplanes))
            layers.append(nn.ReLU(inplace=True))
        self.cls_head = nn.Sequential(*layers)
        self.cls_out = nn.Conv2d(inplanes,
                                 num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.center_out = nn.Conv2d(inplanes,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_out.bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)
        cls_output = self.cls_out(x)
        center_output = self.center_out(x)

        return cls_output, center_output

class FCOSRegHead(nn.Module):
    def __init__(self, inplanes, num_layers=4):
        super(FCOSRegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
            layers.append(nn.GroupNorm(32, inplanes))
            layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*layers)
        self.reg_out = nn.Conv2d(inplanes,
                                 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.reg_head(x)
        reg_output = self.reg_out(x)
        # center_output = self.center_out(x)
        # return reg_output, center_output

        return reg_output

class FCOSPositions(nn.Module):
    def __init__(self, strides):
        super(FCOSPositions, self).__init__()
        self.strides = strides

    def forward(self, batch_size, fpn_feature_sizes):
        """
        generate batch positions
        """
        device = fpn_feature_sizes.device

        one_sample_positions = []
        # 遍历每层的步长和特征图大小
        for stride, fpn_feature_size in zip(self.strides, fpn_feature_sizes):
            # 在单层特征图上，产生特征点坐标
            featrue_positions = self.generate_positions_on_feature_map(
                fpn_feature_size, stride)
            featrue_positions = featrue_positions.to(device)
            one_sample_positions.append(featrue_positions)

        batch_positions = []
        # 遍历每层特征点坐标
        for per_level_featrue_positions in one_sample_positions:
            # unsqueeze(0)表示在第一个维度增加一个维度，这个维度是留给batch size的
            # repeat(batch_size, 1, 1, 1)意思是第一个维度重复batch size遍，其他维度不变（重复一遍）
            per_level_featrue_positions = per_level_featrue_positions.unsqueeze(
                0).repeat(batch_size, 1, 1, 1)
            batch_positions.append(per_level_featrue_positions)

        # if input size:[B,3,640,640]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]
        # per position format:[x_center,y_center]
        return batch_positions

    def generate_positions_on_feature_map(self, feature_map_size, stride):
        """
        generate all positions on a feature map
        """

        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (torch.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (torch.arange(0, feature_map_size[1]) + 0.5) * stride

        # feature_map_positions shape:[w,h,2] -> [h,w,2] -> [h*w,2]
        feature_map_positions = torch.tensor([[[shift_x, shift_y]
                                               for shift_y in shifts_y]
                                              for shift_x in shifts_x
                                              ]).permute(1, 0, 2).contiguous()

        # feature_map_positions format: [point_nums,2],2:[x_center,y_center]
        return feature_map_positions

