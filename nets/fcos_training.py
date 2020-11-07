import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 100000000
class FCOSLoss(nn.Module):
    def __init__(self,
                 strides=[8, 16, 32, 64, 128],
                 mi=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]],
                 alpha=0.25,
                 gamma=2.,
                 epsilon=1e-4):
        super(FCOSLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.strides = strides
        self.mi = mi

    def forward(self, cls_heads, reg_heads, center_heads, batch_positions, annotations, cuda=True):
        """
        compute cls loss, reg loss and center-ness loss in one batch
        """
        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_heads shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        # center_heads shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]
        # annotations shape: [GT个数，5] 例如[[262., 210., 460., 490.,  14.], [295., 305., 460., 471.,  11.]])
        cls_preds, reg_preds, center_preds, batch_targets = self.get_batch_position_annotations(
            cls_heads, reg_heads, center_heads, batch_positions, annotations, cuda=cuda)

        # batch_targets shape:[batch_size, points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        # cls_preds shape:[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, C]
        # reg_preds shape:[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 4]
        # center_preds shape:[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 1]
        cls_preds = torch.sigmoid(cls_preds)
        reg_preds = torch.exp(reg_preds)
        center_preds = torch.sigmoid(center_preds) # ?????????????????????????????????????????????????????????????????????
        batch_targets[:, :, 5:6] = torch.sigmoid(batch_targets[:, :, 5:6])# ?????????????????????????????????????????????????????????????????????

        # device = annotations.device
        cls_loss, reg_loss, center_ness_loss = [], [], []
        valid_image_num = 0
        for per_image_cls_preds, per_image_reg_preds, per_image_center_preds, per_image_targets in zip(
                cls_preds, reg_preds, center_preds, batch_targets):
            # per_image_targets shape:[points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
            # 取每张图l,t,r,b都非0的（正样本）数量
            positive_points_num = (per_image_targets[per_image_targets[:, 4] > 0]).shape[0]
            if positive_points_num == 0:
                if cuda:
                    cls_loss.append(torch.tensor(0.).cuda())
                    reg_loss.append(torch.tensor(0.).cuda())
                    center_ness_loss.append(torch.tensor(0.).cuda())
                else:
                    cls_loss.append(torch.tensor(0.))
                    reg_loss.append(torch.tensor(0.))
                    center_ness_loss.append(torch.tensor(0.))
            else:
                valid_image_num += 1
                one_image_cls_loss = self.compute_one_image_focal_loss(
                    per_image_cls_preds, per_image_targets)
                one_image_reg_loss = self.compute_one_image_giou_loss(
                    per_image_reg_preds, per_image_targets, cuda)
                one_image_center_ness_loss = self.compute_one_image_center_ness_loss(
                    per_image_center_preds, per_image_targets, cuda)

                cls_loss.append(one_image_cls_loss)
                reg_loss.append(one_image_reg_loss)
                center_ness_loss.append(one_image_center_ness_loss)

        # 此举就是避免分母为0
        valid_image_num = 1 if valid_image_num == 0 else valid_image_num
        # 求平均loss
        cls_loss = sum(cls_loss) / valid_image_num
        reg_loss = sum(reg_loss) / valid_image_num
        center_ness_loss = sum(center_ness_loss) / valid_image_num

        return cls_loss, reg_loss, center_ness_loss

    def compute_one_image_focal_loss(self, per_image_cls_preds,
                                     per_image_targets):
        """
        compute one image focal loss(cls loss)
        per_image_cls_preds:[points_num,num_classes]
        per_image_targets:[points_num,8]
        """
        per_image_cls_preds = torch.clamp(per_image_cls_preds,
                                          min=self.epsilon,
                                          max=1. - self.epsilon)
        num_classes = per_image_cls_preds.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(per_image_targets[:, 4].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(per_image_cls_preds) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), per_image_cls_preds,
                         1. - per_image_cls_preds)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        bce_loss = -(
            loss_ground_truth * torch.log(per_image_cls_preds) +
            (1. - loss_ground_truth) * torch.log(1. - per_image_cls_preds))

        one_image_focal_loss = focal_weight * bce_loss

        one_image_focal_loss = one_image_focal_loss.sum()
        positive_points_num = per_image_targets[
            per_image_targets[:, 4] > 0].shape[0]
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        one_image_focal_loss = one_image_focal_loss / positive_points_num

        return one_image_focal_loss

    def compute_one_image_giou_loss(self, per_image_reg_preds,
                                    per_image_targets, cuda):
        """
        compute one image giou loss(reg loss)
        per_image_reg_preds:[points_num,4]
        per_image_targets:[anchor_num,8]
        """
        # only use positive points sample to compute reg loss
        # device = per_image_reg_preds.device
        per_image_reg_preds = per_image_reg_preds[per_image_targets[:, 4] > 0]
        per_image_targets = per_image_targets[per_image_targets[:, 4] > 0]
        positive_points_num = per_image_targets.shape[0]

        if positive_points_num == 0:
            if cuda:
                return torch.tensor(0.).cuda()
            else:
                return torch.tensor(0.)

        center_ness_targets = per_image_targets[:, 5]

        pred_bboxes_xy_min = per_image_targets[:,
                                               6:8] - per_image_reg_preds[:,
                                                                          0:2]
        pred_bboxes_xy_max = per_image_targets[:,
                                               6:8] + per_image_reg_preds[:,
                                                                          2:4]
        gt_bboxes_xy_min = per_image_targets[:, 6:8] - per_image_targets[:,
                                                                         0:2]
        gt_bboxes_xy_max = per_image_targets[:, 6:8] + per_image_targets[:,
                                                                         2:4]

        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                axis=1)
        gt_bboxes = torch.cat([gt_bboxes_xy_min, gt_bboxes_xy_max], axis=1)

        overlap_area_top_left = torch.max(pred_bboxes[:, 0:2], gt_bboxes[:,
                                                                         0:2])
        overlap_area_bot_right = torch.min(pred_bboxes[:, 2:4], gt_bboxes[:,
                                                                          2:4])
        overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                         overlap_area_top_left,
                                         min=0)
        overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

        # anchors and annotations convert format to [x1,y1,w,h]
        pred_bboxes_w_h = pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1
        gt_bboxes_w_h = gt_bboxes[:, 2:4] - gt_bboxes[:, 0:2] + 1

        # compute anchors_area and annotations_area
        pred_bboxes_area = pred_bboxes_w_h[:, 0] * pred_bboxes_w_h[:, 1]
        gt_bboxes_area = gt_bboxes_w_h[:, 0] * gt_bboxes_w_h[:, 1]

        # compute union_area
        union_area = pred_bboxes_area + gt_bboxes_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        # compute ious between one image anchors and one image annotations
        ious = overlap_area / union_area

        enclose_area_top_left = torch.min(pred_bboxes[:, 0:2], gt_bboxes[:,
                                                                         0:2])
        enclose_area_bot_right = torch.max(pred_bboxes[:, 2:4], gt_bboxes[:,
                                                                          2:4])
        enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                         enclose_area_top_left,
                                         min=0)
        enclose_area = enclose_area_sizes[:, 0] * enclose_area_sizes[:, 1]
        enclose_area = torch.clamp(enclose_area, min=1e-4)

        gious_loss = 1. - ious + (enclose_area - union_area) / enclose_area
        gious_loss = torch.clamp(gious_loss, min=-1.0, max=1.0)
        # use center_ness_targets as the weight of gious loss
        gious_loss = gious_loss * center_ness_targets
        gious_loss = gious_loss.sum() / positive_points_num
        gious_loss = 2. * gious_loss

        return gious_loss

    def compute_one_image_center_ness_loss(self, per_image_center_preds,
                                           per_image_targets, cuda):
        """
        compute one image center_ness loss(center ness loss)
        per_image_center_preds:[points_num,4]
        per_image_targets:[anchor_num,8]
        """
        # only use positive points sample to compute center_ness loss
        # device = per_image_center_preds.device
        per_image_center_preds = per_image_center_preds[
            per_image_targets[:, 4] > 0]
        per_image_targets = per_image_targets[per_image_targets[:, 4] > 0]
        positive_points_num = per_image_targets.shape[0]

        if positive_points_num == 0:
            if cuda:
                return torch.tensor(0.).cuda()
            else:
                return torch.tensor(0.)

        center_ness_targets = per_image_targets[:, 5:6]

        center_ness_loss = -(
            center_ness_targets * torch.log(per_image_center_preds) +
            (1. - center_ness_targets) *
            torch.log(1. - per_image_center_preds))
        center_ness_loss = center_ness_loss.sum() / positive_points_num

        return center_ness_loss

    def get_batch_position_annotations(self, cls_heads, reg_heads,
                                       center_heads, batch_positions,
                                       annotations, cuda):
        """
        Assign a ground truth target for each position on feature map
        """
        # device = annotations.device
        batch_mi = []
        # 遍历每层的reg_head的输出和最大回归距离mi
        for reg_head, mi in zip(reg_heads, self.mi):
            if cuda:
                mi = torch.tensor(mi).cuda()
                B, H, W, _ = reg_head.shape
                per_level_mi = torch.zeros(B, H, W, 2).cuda()
            else:
                mi = torch.tensor(mi)
                B, H, W, _ = reg_head.shape
                per_level_mi = torch.zeros(B, H, W, 2)
            per_level_mi = per_level_mi + mi
            # batch_mi shape: [B, H, W, 2]，这个2就是self.mi里面，每一个mi对应的上下区间
            batch_mi.append(per_level_mi)

        cls_preds,reg_preds,center_preds,all_points_position,all_points_mi=[],[],[],[],[]
        # 遍历每层的分类、回归、centerness、位置、mi
        for cls_pred, reg_pred, center_pred, per_level_position, per_level_mi in zip(
                cls_heads, reg_heads, center_heads, batch_positions, batch_mi):
            # [B, H, W, C] -> [B, H×W, C]
            cls_pred = cls_pred.view(cls_pred.shape[0], -1, cls_pred.shape[-1])
            # [B, H, W, 4] -> [B, H×W, 4]
            reg_pred = reg_pred.view(reg_pred.shape[0], -1, reg_pred.shape[-1])
            # [B, H, W, 1] -> [B, H×W, 1]
            center_pred = center_pred.view(center_pred.shape[0], -1,
                                           center_pred.shape[-1])
            # [B, H, W, 2] -> [B, H×W, 2]
            per_level_position = per_level_position.view(
                per_level_position.shape[0], -1, per_level_position.shape[-1])
            # [B, H, W, 2] -> [B, H×W, 2]
            per_level_mi = per_level_mi.view(per_level_mi.shape[0], -1,
                                             per_level_mi.shape[-1])
            # cls_preds为装有5层cls_pred的list
            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            center_preds.append(center_pred)
            all_points_position.append(per_level_position)
            all_points_mi.append(per_level_mi)

        # axis=1按列拼接，由原来5个层的list，变为[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 20]
        cls_preds = torch.cat(cls_preds, axis=1)
        # axis=1按列拼接，由原来5个层的list，变为[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 4]
        reg_preds = torch.cat(reg_preds, axis=1)
        # axis=1按列拼接，由原来5个层的list，变为[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 1]
        center_preds = torch.cat(center_preds, axis=1)
        # axis=1按列拼接，由原来5个层的list，变为[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 2]
        all_points_position = torch.cat(all_points_position, axis=1)
        # axis=1按列拼接，由原来5个层的list，变为[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 2]
        all_points_mi = torch.cat(all_points_mi, axis=1)
        # 至此，就消除了各个特征层的概念了，把所有的层都混一起了

        batch_targets = []
        # 遍历每个batch size中的每个图片的上的点、mi、GT标注
        for per_image_position, per_image_mi, per_image_annotations in zip(
                all_points_position, all_points_mi, annotations):
            if per_image_annotations.shape[0] != 0:
                # 筛选GT对应的分类>=0的GT
                per_image_annotations = per_image_annotations[
                    per_image_annotations[:, 4] >= 0]
            # 这张图片上坐标点的数量
            points_num = per_image_position.shape[0]

            if per_image_annotations.shape[0] == 0:
                if cuda:
                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6]).cuda()
                else:
                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6])
            else:
                # 这张图片上GT的数量
                annotaion_num = per_image_annotations.shape[0]
                # 这张图片各个GT的坐标点（刨去分类），shape:[GT数量,4]
                per_image_gt_bboxes = per_image_annotations[:, 0:4]
                if cuda:
                    candidates = torch.zeros([points_num, annotaion_num, 4]).cuda()
                else:
                    candidates = torch.zeros([points_num, annotaion_num, 4])
                # candidates shape:[坐标点数量, GT数量, 4]，他就是坐标点数量个，GT数量行4列的一个张量，GT数量行4列数据描述的就是GT坐标
                candidates = candidates + per_image_gt_bboxes.unsqueeze(0)
                # per_image_position：[H1×W1+H2×W2+H3×W3+H4×W4+H5×W5,2] -> [H1×W1+H2×W2+H3×W3+H4×W4+H5×W5,1,2] -> [H1×W1+H2×W2+H3×W3+H4×W4+H5×W5,GT数量,4]
                # per_image_position shape：[坐标点数量, GT数量, 4]，他就是坐标点数量个，GT数量行4列的一个张量，GT数量行4列数据是重复的中心点坐标
                per_image_position = per_image_position.unsqueeze(1).repeat(1, annotaion_num, 2)

                # 中心点坐标距离左边和上边的距离：l,t
                # candidates shape:[坐标点数量, GT数量, 4]
                candidates[:, :, 0:2] = per_image_position[:, :, 0:2] - candidates[:, :, 0:2]
                # 中心点坐标距离右边和下边的距离：r,b
                candidates[:, :, 2:4] = candidates[:, :, 2:4] - per_image_position[:, :, 2:4]
                # 取最后一维（就是l,t,r,b）的最小值，keepdims主要用于保持矩阵的二维特性
                # candidates_min_value shape:[坐标点数量, GT数量, 1]
                candidates_min_value, _ = candidates.min(axis=-1, keepdim=True)
                # sample_flag shape:[坐标点数量, GT数量, 1]，判断如果min(l,t,r,b)>0，那就是落在GT中，此时将candidates_min_value中的负数变为0，正数变为1，维度不变，赋值给sample_flag
                sample_flag = (candidates_min_value[:, :, 0] > 0).int().unsqueeze(-1)
                # get all negative reg targets which points ctr out of gt box
                # 这样一乘，就消除了上面candidates坐标中所有的负数
                candidates = candidates * sample_flag

                # get all negative reg targets which assign ground turth not in range of mi
                # 取最后一维（就是l,t,r,b）的最大值，keepdims主要用于保持矩阵的二维特性
                # candidates_max_value shape:[坐标点数量, GT数量, 1]
                candidates_max_value, _ = candidates.max(axis=-1, keepdim=True)
                # per_image_mi shape: [单张图片中心点个数,2] -> [单张图片中心点个数,GT个数,2]
                per_image_mi = per_image_mi.unsqueeze(1).repeat( 1, annotaion_num, 1)
                # 判断是否max(l∗, t∗, r∗, b∗) > mi，是的话原位置为1，否的话原位置为0，维度不变的把这个结果赋值给m1_negative_flag
                m1_negative_flag = (candidates_max_value[:, :, 0] > per_image_mi[:, :, 0]).int().unsqueeze(-1)
                candidates = candidates * m1_negative_flag

                # 判断是否max(l∗, t∗, r∗, b∗) < mi-1，是的话原位置为1，否的话原位置为0，维度不变的把这个结果赋值给m2_negative_flag
                m2_negative_flag = (candidates_max_value[:, :, 0] < per_image_mi[:, :, 1]).int().unsqueeze(-1)
                candidates = candidates * m2_negative_flag

                # 至此，除了ambiguous sample，各个sample都分到了响应的FPN层

                # candidates shape:[坐标点数量, GT数量, 4]
                # 判断这张图片中的中心点，不管对应哪个GT，是否在GT内(通过四个距离的和判断是否大于0得出)，是的话，该中心点坐标对应的那个地方为True，否则为False
                # final_sample_flag shape: [中心点数量]
                final_sample_flag = candidates.sum(axis=-1).sum(axis=-1)
                final_sample_flag = final_sample_flag > 0
                # nonzero在此处的一维数组里，返回元素值== True的下标；squeeze(dim=-1)是扒掉最外一维。整句就是返回元素值== True的下标
                # positive_index shape: [中心点数量]
                positive_index = (final_sample_flag == True).nonzero().squeeze(dim=-1)

                # if no assign positive sample
                if len(positive_index) == 0:
                    del candidates
                    if cuda:
                        # 6:l,t,r,b,class_index,center-ness_gt
                        per_image_targets = torch.zeros([points_num, 6]).cuda()
                    else:
                        # 6:l,t,r,b,class_index,center-ness_gt
                        per_image_targets = torch.zeros([points_num, 6])
                else:
                    # candidates shape:[坐标点数量, GT数量, 4]
                    # positive_index shape: [正样本坐标点数量]
                    # positive_candidates shape: [正样本坐标点数量, GT数量, 4]
                    positive_candidates = candidates[positive_index]

                    del candidates

                    # per_image_annotations shape:[GT数量,5]
                    # sample_box_gts shape:[1,GT数量,4]，这里的4是抛开了类别的GT的两个角点坐标
                    sample_box_gts = per_image_annotations[:, 0:4].unsqueeze(0)
                    # sample_box_gts shape:[正样本坐标点数量,GT数量,4]
                    sample_box_gts = sample_box_gts.repeat(positive_candidates.shape[0], 1, 1)

                    # per_image_annotations shape:[GT数量,5]
                    # sample_class_gts shape:[1,GT数量,1]，这里的1是抛开了坐标
                    sample_class_gts = per_image_annotations[:, 4].unsqueeze(-1).unsqueeze(0)
                    # sample_class_gts shape:[正样本坐标点数量,GT数量,1]
                    sample_class_gts = sample_class_gts.repeat(positive_candidates.shape[0], 1, 1)

                    if cuda:
                        # 6:l,t,r,b,class_index,center-ness_gt
                        per_image_targets = torch.zeros([points_num, 6]).cuda()
                    else:
                        # 6:l,t,r,b,class_index,center-ness_gt
                        per_image_targets = torch.zeros([points_num, 6])

                    # positive_candidates shape: [正样本坐标点数量, GT数量, 4]
                    # 如果positive_candidates的第二维GT数量对应一个GT
                    if positive_candidates.shape[1] == 1:
                        # if only one candidate for each positive sample
                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        # class_index value from 1 to 80 represent 80 positive classes
                        # class_index value 0 represenet negative class
                        # positive_candidates shape: [正样本坐标点数量, GT数量(1), 4] -> [正样本坐标点数量, 4]
                        positive_candidates = positive_candidates.squeeze(1)
                        # sample_class_gts shape:[正样本坐标点数量,GT数量(1),1] -> [正样本坐标点数量, 1]
                        sample_class_gts = sample_class_gts.squeeze(1)
                        # positive_index shape: [中心点数量]
                        # per_image_targets shape: [中心点数量,6]
                        # per_image_targets中负样本的地方就全是0了
                        per_image_targets[positive_index, 0:4] = positive_candidates
                        per_image_targets[positive_index, 4:5] = sample_class_gts + 1

                        l, t, r, b = per_image_targets[positive_index, 0:1], per_image_targets[positive_index, 1:2], per_image_targets[positive_index,2:3], per_image_targets[positive_index, 3:4]
                        # per_image_targets shape: [中心点数量,6]，这里6代表l,t,r,b,class_index,center-ness_gt
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))
                    else:
                        # if a positive point sample have serveral object candidates,then choose the smallest area object candidate as the ground turth for this positive point sample
                        # sample_box_gts shape:[正样本坐标点数量,GT数量,4]，通过将GT角点右下角坐标减去左上角坐标得到框的宽高
                        gts_w_h = sample_box_gts[:, :, 2:4] - sample_box_gts[:, :, 0:2]
                        # 宽×高 得到框的区域的面积
                        gts_area = gts_w_h[:, :, 0] * gts_w_h[:, :, 1]
                        # positive_candidates shape: [正样本坐标点数量, GT数量, 4]
                        # positive_candidates shape: [正样本坐标点数量, GT数量]
                        positive_candidates_value = positive_candidates.sum(axis=2)

                        # make sure all negative candidates areas==100000000,thus .min() operation wouldn't choose negative candidates
                        # 这句话没必要懂
                        INF = 100000000
                        inf_tensor = torch.ones_like(gts_area) * INF
                        gts_area = torch.where(torch.eq(positive_candidates_value, 0.), inf_tensor, gts_area)

                        # get the smallest object candidate index
                        _, min_index = gts_area.min(axis=1)
                        candidate_indexes = (torch.linspace(1, positive_candidates.shape[0], positive_candidates.shape[0]) - 1).long()
                        final_candidate_reg_gts = positive_candidates[candidate_indexes, min_index, :]
                        final_candidate_cls_gts = sample_class_gts[candidate_indexes, min_index]

                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        per_image_targets[positive_index, 0:4] = final_candidate_reg_gts
                        per_image_targets[positive_index, 4:5] = final_candidate_cls_gts + 1

                        l, t, r, b = per_image_targets[positive_index, 0:1], per_image_targets[positive_index, 1:2], per_image_targets[positive_index, 2:3], per_image_targets[positive_index, 3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))

            per_image_targets = per_image_targets.unsqueeze(0)
            # batch_targets shape: [batch size,中心点数量,6]
            batch_targets.append(per_image_targets)

        batch_targets = torch.cat(batch_targets, axis=0)
        # 把每个中心点坐标又加进来了
        batch_targets = torch.cat([batch_targets, all_points_position], axis=2)

        # batch_targets shape:[batch_size, points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        # cls_preds shape:[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, C]
        # reg_preds shape:[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 4]
        # center_preds shape:[B, H1×W1+H2×W2+H3×W3+H4×W4+H5×W5, 1]
        return cls_preds, reg_preds, center_preds, batch_targets
