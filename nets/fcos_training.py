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

    def forward(self, cls_heads, reg_heads, center_heads, batch_positions,
                annotations):
        """
        compute cls loss, reg loss and center-ness loss in one batch
        """
        cls_preds, reg_preds, center_preds, batch_targets = self.get_batch_position_annotations(
            cls_heads, reg_heads, center_heads, batch_positions, annotations)

        cls_preds = torch.sigmoid(cls_preds)
        reg_preds = torch.exp(reg_preds)
        center_preds = torch.sigmoid(center_preds)
        batch_targets[:, :, 5:6] = torch.sigmoid(batch_targets[:, :, 5:6])

        device = annotations.device
        cls_loss, reg_loss, center_ness_loss = [], [], []
        valid_image_num = 0
        for per_image_cls_preds, per_image_reg_preds, per_image_center_preds, per_image_targets in zip(
                cls_preds, reg_preds, center_preds, batch_targets):
            positive_points_num = (
                per_image_targets[per_image_targets[:, 4] > 0]).shape[0]
            if positive_points_num == 0:
                cls_loss.append(torch.tensor(0.).to(device))
                reg_loss.append(torch.tensor(0.).to(device))
                center_ness_loss.append(torch.tensor(0.).to(device))
            else:
                valid_image_num += 1
                one_image_cls_loss = self.compute_one_image_focal_loss(
                    per_image_cls_preds, per_image_targets)
                one_image_reg_loss = self.compute_one_image_giou_loss(
                    per_image_reg_preds, per_image_targets)
                one_image_center_ness_loss = self.compute_one_image_center_ness_loss(
                    per_image_center_preds, per_image_targets)

                cls_loss.append(one_image_cls_loss)
                reg_loss.append(one_image_reg_loss)
                center_ness_loss.append(one_image_center_ness_loss)

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
                                    per_image_targets):
        """
        compute one image giou loss(reg loss)
        per_image_reg_preds:[points_num,4]
        per_image_targets:[anchor_num,8]
        """
        # only use positive points sample to compute reg loss
        device = per_image_reg_preds.device
        per_image_reg_preds = per_image_reg_preds[per_image_targets[:, 4] > 0]
        per_image_targets = per_image_targets[per_image_targets[:, 4] > 0]
        positive_points_num = per_image_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

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
                                           per_image_targets):
        """
        compute one image center_ness loss(center ness loss)
        per_image_center_preds:[points_num,4]
        per_image_targets:[anchor_num,8]
        """
        # only use positive points sample to compute center_ness loss
        device = per_image_center_preds.device
        per_image_center_preds = per_image_center_preds[
            per_image_targets[:, 4] > 0]
        per_image_targets = per_image_targets[per_image_targets[:, 4] > 0]
        positive_points_num = per_image_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0.).to(device)

        center_ness_targets = per_image_targets[:, 5:6]

        center_ness_loss = -(
            center_ness_targets * torch.log(per_image_center_preds) +
            (1. - center_ness_targets) *
            torch.log(1. - per_image_center_preds))
        center_ness_loss = center_ness_loss.sum() / positive_points_num

        return center_ness_loss

    def get_batch_position_annotations(self, cls_heads, reg_heads,
                                       center_heads, batch_positions,
                                       annotations):
        """
        Assign a ground truth target for each position on feature map
        """
        device = annotations.device
        batch_mi = []
        for reg_head, mi in zip(reg_heads, self.mi):
            mi = torch.tensor(mi).to(device)
            B, H, W, _ = reg_head.shape
            per_level_mi = torch.zeros(B, H, W, 2).to(device)
            per_level_mi = per_level_mi + mi
            batch_mi.append(per_level_mi)

        cls_preds,reg_preds,center_preds,all_points_position,all_points_mi=[],[],[],[],[]
        for cls_pred, reg_pred, center_pred, per_level_position, per_level_mi in zip(
                cls_heads, reg_heads, center_heads, batch_positions, batch_mi):
            cls_pred = cls_pred.view(cls_pred.shape[0], -1, cls_pred.shape[-1])
            reg_pred = reg_pred.view(reg_pred.shape[0], -1, reg_pred.shape[-1])
            center_pred = center_pred.view(center_pred.shape[0], -1,
                                           center_pred.shape[-1])
            per_level_position = per_level_position.view(
                per_level_position.shape[0], -1, per_level_position.shape[-1])
            per_level_mi = per_level_mi.view(per_level_mi.shape[0], -1,
                                             per_level_mi.shape[-1])

            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            center_preds.append(center_pred)
            all_points_position.append(per_level_position)
            all_points_mi.append(per_level_mi)

        cls_preds = torch.cat(cls_preds, axis=1)
        reg_preds = torch.cat(reg_preds, axis=1)
        center_preds = torch.cat(center_preds, axis=1)
        all_points_position = torch.cat(all_points_position, axis=1)
        all_points_mi = torch.cat(all_points_mi, axis=1)

        batch_targets = []
        for per_image_position, per_image_mi, per_image_annotations in zip(
                all_points_position, all_points_mi, annotations):
            per_image_annotations = per_image_annotations[
                per_image_annotations[:, 4] >= 0]
            points_num = per_image_position.shape[0]

            if per_image_annotations.shape[0] == 0:
                # 6:l,t,r,b,class_index,center-ness_gt
                per_image_targets = torch.zeros([points_num, 6], device=device)
            else:
                annotaion_num = per_image_annotations.shape[0]
                per_image_gt_bboxes = per_image_annotations[:, 0:4]
                candidates = torch.zeros([points_num, annotaion_num, 4],
                                         device=device)
                candidates = candidates + per_image_gt_bboxes.unsqueeze(0)
                per_image_position = per_image_position.unsqueeze(1).repeat(
                    1, annotaion_num, 2)
                candidates[:, :,
                           0:2] = per_image_position[:, :,
                                                     0:2] - candidates[:, :,
                                                                       0:2]
                candidates[:, :,
                           2:4] = candidates[:, :,
                                             2:4] - per_image_position[:, :,
                                                                       2:4]

                candidates_min_value, _ = candidates.min(axis=-1, keepdim=True)
                sample_flag = (candidates_min_value[:, :, 0] >
                               0).int().unsqueeze(-1)
                # get all negative reg targets which points ctr out of gt box
                candidates = candidates * sample_flag

                # get all negative reg targets which assign ground turth not in range of mi
                candidates_max_value, _ = candidates.max(axis=-1, keepdim=True)
                per_image_mi = per_image_mi.unsqueeze(1).repeat(
                    1, annotaion_num, 1)
                m1_negative_flag = (candidates_max_value[:, :, 0] >
                                    per_image_mi[:, :, 0]).int().unsqueeze(-1)
                candidates = candidates * m1_negative_flag
                m2_negative_flag = (candidates_max_value[:, :, 0] <
                                    per_image_mi[:, :, 1]).int().unsqueeze(-1)
                candidates = candidates * m2_negative_flag

                final_sample_flag = candidates.sum(axis=-1).sum(axis=-1)
                final_sample_flag = final_sample_flag > 0
                positive_index = (final_sample_flag == True).nonzero().squeeze(
                    dim=-1)

                # if no assign positive sample
                if len(positive_index) == 0:
                    del candidates
                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6],
                                                    device=device)
                else:
                    positive_candidates = candidates[positive_index]

                    del candidates

                    sample_box_gts = per_image_annotations[:, 0:4].unsqueeze(0)
                    sample_box_gts = sample_box_gts.repeat(
                        positive_candidates.shape[0], 1, 1)
                    sample_class_gts = per_image_annotations[:, 4].unsqueeze(
                        -1).unsqueeze(0)
                    sample_class_gts = sample_class_gts.repeat(
                        positive_candidates.shape[0], 1, 1)

                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6],
                                                    device=device)

                    if positive_candidates.shape[1] == 1:
                        # if only one candidate for each positive sample
                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        # class_index value from 1 to 80 represent 80 positive classes
                        # class_index value 0 represenet negative class
                        positive_candidates = positive_candidates.squeeze(1)
                        sample_class_gts = sample_class_gts.squeeze(1)
                        per_image_targets[positive_index,
                                          0:4] = positive_candidates
                        per_image_targets[positive_index,
                                          4:5] = sample_class_gts + 1

                        l, t, r, b = per_image_targets[
                            positive_index, 0:1], per_image_targets[
                                positive_index, 1:2], per_image_targets[
                                    positive_index,
                                    2:3], per_image_targets[positive_index,
                                                            3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))
                    else:
                        # if a positive point sample have serveral object candidates,then choose the smallest area object candidate as the ground turth for this positive point sample
                        gts_w_h = sample_box_gts[:, :,
                                                 2:4] - sample_box_gts[:, :,
                                                                       0:2]
                        gts_area = gts_w_h[:, :, 0] * gts_w_h[:, :, 1]
                        positive_candidates_value = positive_candidates.sum(
                            axis=2)

                        # make sure all negative candidates areas==100000000,thus .min() operation wouldn't choose negative candidates
                        INF = 100000000
                        inf_tensor = torch.ones_like(gts_area) * INF
                        gts_area = torch.where(
                            torch.eq(positive_candidates_value, 0.),
                            inf_tensor, gts_area)

                        # get the smallest object candidate index
                        _, min_index = gts_area.min(axis=1)
                        candidate_indexes = (
                            torch.linspace(1, positive_candidates.shape[0],
                                           positive_candidates.shape[0]) -
                            1).long()
                        final_candidate_reg_gts = positive_candidates[
                            candidate_indexes, min_index, :]
                        final_candidate_cls_gts = sample_class_gts[
                            candidate_indexes, min_index]

                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        per_image_targets[positive_index,
                                          0:4] = final_candidate_reg_gts
                        per_image_targets[positive_index,
                                          4:5] = final_candidate_cls_gts + 1

                        l, t, r, b = per_image_targets[
                            positive_index, 0:1], per_image_targets[
                                positive_index, 1:2], per_image_targets[
                                    positive_index,
                                    2:3], per_image_targets[positive_index,
                                                            3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))

            per_image_targets = per_image_targets.unsqueeze(0)
            batch_targets.append(per_image_targets)

        batch_targets = torch.cat(batch_targets, axis=0)
        batch_targets = torch.cat([batch_targets, all_points_position], axis=2)

        # batch_targets shape:[batch_size, points_num, 8],8:l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        return cls_preds, reg_preds, center_preds, batch_targets
