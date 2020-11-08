import torch
import torch.nn as nn
from torchvision.ops import nms



class FCOSDecoder(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 strides=[8, 16, 32, 64, 128],
                 top_n=1000,
                 min_score_threshold=0.01,
                 nms_threshold=0.6,
                 max_detection_num=100):
        super(FCOSDecoder, self).__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.strides = strides
        self.top_n = top_n
        self.min_score_threshold = min_score_threshold
        self.nms_threshold = nms_threshold
        self.max_detection_num = max_detection_num

    def forward(self, cls_heads, reg_heads, center_heads, batch_positions):
        with torch.no_grad():
            device = cls_heads[0].device

            filter_scores,filter_score_classes,filter_reg_heads,filter_cls_heads,filter_batch_positions=[],[],[],[],[]
            for per_level_cls_head, per_level_reg_head, per_level_center_head, per_level_position in zip(
                    cls_heads, reg_heads, center_heads, batch_positions):
                per_level_cls_head = torch.sigmoid(per_level_cls_head)
                per_level_reg_head = torch.exp(per_level_reg_head)
                per_level_center_head = torch.sigmoid(per_level_center_head)

                # [B, H, W, C] -> [B, H×W, C]
                per_level_cls_head = per_level_cls_head.view(per_level_cls_head.shape[0], -1, per_level_cls_head.shape[-1])
                # [B, H, W, 4] -> [B, H×W, 4]
                per_level_reg_head = per_level_reg_head.view(per_level_reg_head.shape[0], -1, per_level_reg_head.shape[-1])
                # [B, H, W, 1] -> [B, H×W, 1]
                per_level_center_head = per_level_center_head.view(per_level_center_head.shape[0], -1, per_level_center_head.shape[-1])
                # [B, H, W, 2] -> [B, H×W, 2]
                per_level_position = per_level_position.view(per_level_position.shape[0], -1, per_level_position.shape[-1])
                # 返回本层每个点对应的分数最大值和对应的位置（也就是分类编号，即分类）
                # score_classes是每个点的C个类别里面，分值最大的那个分数的下标
                # scores, score_classes shape: [B, H×W]
                scores, score_classes = torch.max(per_level_cls_head, dim=2)
                # 此分数与centerness相乘再开方得到新的分数
                # scores shape: [B, H×W]
                scores = torch.sqrt(scores * per_level_center_head.squeeze(-1))
                if scores.shape[1] >= self.top_n:
                    # 取分数降序的top_n，以及对应的下标
                    # indexes是分数在top_n里面的点的下标（第几个点分值在top_n里）
                    # scores、indexes shape: [B, self.top_n]
                    scores, indexes = torch.topk(scores, self.top_n, dim=1, largest=True, sorted=True)
                    # 返回本层每个点对应的分数最大值top_n，对应的位置（也就是分类编号，即分类）
                    # score_classes shape: [B, self.top_n]
                    score_classes = torch.gather(score_classes, 1, indexes)

                    # per_level_reg_head shape：[B, H×W, 4] -> [B, self.top_n, 4]
                    # indexes.unsqueeze(-1).repeat(1, 1, 4) shape：[B, self.top_n, 4]
                    # 简单理解就是将本层每个点的回归预测，缩至为，得分top_n的点的回归预测
                    per_level_reg_head = torch.gather(per_level_reg_head, 1, indexes.unsqueeze(-1).repeat(1, 1, 4))
                    # per_level_center_head shape：[B, H×W, 4] -> [B, self.top_n, 1]
                    # 简单理解就是将本层每个点的centerness预测，缩至为，得分top_n的点的centerness预测
                    per_level_center_head = torch.gather(per_level_center_head, 1,indexes.unsqueeze(-1).repeat(1, 1, 1))
                    # per_level_position shape：[B, H×W, 4] -> [B, self.top_n, 2]
                    # 简单理解就是将本层每个点，缩至为，得分top_n的点
                    per_level_position = torch.gather(per_level_position, 1, indexes.unsqueeze(-1).repeat(1, 1, 2))

                    per_level_cls_head = torch.gather(per_level_cls_head, 1, indexes.unsqueeze(-1).repeat(1, 1, per_level_cls_head.shape[-1]))

                filter_scores.append(scores)
                filter_score_classes.append(score_classes)
                filter_reg_heads.append(per_level_reg_head)
                filter_batch_positions.append(per_level_position)
                filter_cls_heads.append(per_level_cls_head)

            # 每层叠在一起
            filter_scores = torch.cat(filter_scores, axis=1)
            filter_score_classes = torch.cat(filter_score_classes, axis=1)
            filter_reg_heads = torch.cat(filter_reg_heads, axis=1)
            filter_batch_positions = torch.cat(filter_batch_positions, axis=1)
            filter_cls_heads = torch.cat(filter_cls_heads, axis=1)

            batch_scores, batch_classes, batch_pred_bboxes, batch_pred_cls = [], [], [], []
            # 对每张图top_n的点位、得分、得分下标、回归预测进行遍历
            # scores shape:[每张图片点数]
            # score_classes shape:[每张图片点数]
            # per_image_reg_preds shape:[每张图片点数,4]
            # per_image_points_position shape:[每张图片点数,2]
            # per_image_cls_preds shape:[每张图片点数,C]
            for scores, score_classes, per_image_reg_preds, per_image_points_position, per_image_cls_preds in zip(
                    filter_scores, filter_score_classes, filter_reg_heads, filter_batch_positions, filter_cls_heads):
                # pred_bboxes shape:[每张图片点数,4]
                pred_bboxes = self.snap_ltrb_reg_heads_to_x1_y1_x2_y2_bboxes(
                    per_image_reg_preds, per_image_points_position)

                # 选取大于置信度阈值的预测数据
                score_classes = score_classes[
                    scores > self.min_score_threshold].float()
                pred_bboxes = pred_bboxes[
                    scores > self.min_score_threshold].float()
                per_image_cls_preds = per_image_cls_preds[scores > self.min_score_threshold].float()
                scores = scores[scores > self.min_score_threshold].float()

                one_image_scores = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                one_image_classes = (-1) * torch.ones(
                    (self.max_detection_num, ), device=device)
                one_image_pred_bboxes = (-1) * torch.ones(
                    (self.max_detection_num, 4), device=device)
                one_image_per_image_cls_preds = (-1) * torch.ones(
                    (self.max_detection_num, per_image_cls_preds.shape[-1]), device=device)

                if scores.shape[0] != 0:
                    # Sort boxes
                    sorted_scores, sorted_indexes = torch.sort(scores,
                                                               descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_pred_bboxes = pred_bboxes[sorted_indexes]
                    sorted_per_image_cls_preds = per_image_cls_preds[sorted_indexes]

                    keep = nms(sorted_pred_bboxes, sorted_scores,
                               self.nms_threshold)
                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_pred_bboxes = sorted_pred_bboxes[keep]
                    keep_per_image_cls_preds = sorted_per_image_cls_preds[keep]

                    final_detection_num = min(self.max_detection_num, keep_scores.shape[0])

                    one_image_scores[0:final_detection_num] = keep_scores[0:final_detection_num]
                    one_image_classes[0:final_detection_num] = keep_classes[0:final_detection_num]
                    one_image_pred_bboxes[0:final_detection_num, :] = keep_pred_bboxes[0:final_detection_num, :]
                    one_image_per_image_cls_preds[0:final_detection_num, :] = keep_per_image_cls_preds[0:final_detection_num, :]

                # [1, 经过nms后留下的正样本数]
                one_image_scores = one_image_scores.unsqueeze(0)
                # [1, 经过nms后留下的正样本数]
                one_image_classes = one_image_classes.unsqueeze(0)
                # [1, 经过nms后留下的正样本数, 4]
                one_image_pred_bboxes = one_image_pred_bboxes.unsqueeze(0)
                one_image_per_image_cls_preds = one_image_per_image_cls_preds.unsqueeze(0)

                # 一个装着batch个形如[1, 经过nms后留下的正样本数]list
                batch_scores.append(one_image_scores)
                # 一个装着batch个形如[1, 经过nms后留下的正样本数]list
                batch_classes.append(one_image_classes)
                # 一个装着batch个形如[1, 经过nms后留下的正样本数, 4]list
                batch_pred_bboxes.append(one_image_pred_bboxes)
                batch_pred_cls.append(one_image_per_image_cls_preds)

            # [B, 经过nms后留下的正样本数]
            batch_scores = torch.cat(batch_scores, axis=0)
            # [B, 经过nms后留下的正样本数]
            batch_classes = torch.cat(batch_classes, axis=0)
            # [B, 经过nms后留下的正样本数, 4]
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, axis=0)
            batch_pred_cls = torch.cat(batch_pred_cls, axis=0)

            # batch_scores shape:[batch_size,max_detection_num]
            # batch_classes shape:[batch_size,max_detection_num]
            # batch_pred_bboxes shape[batch_size,max_detection_num,4]
            # batch_pred_cls shape[batch_size,max_detection_num,C]
            return batch_scores, batch_classes, batch_pred_bboxes, batch_pred_cls

    def snap_ltrb_reg_heads_to_x1_y1_x2_y2_bboxes(self, reg_preds,
                                                  points_position):
        """
        snap reg preds to pred bboxes
        reg_preds:[points_num,4],4:[l,t,r,b]
        points_position:[points_num,2],2:[point_ctr_x,point_ctr_y]
        """
        pred_bboxes_xy_min = points_position - reg_preds[:, 0:2]
        pred_bboxes_xy_max = points_position + reg_preds[:, 2:4]
        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max],
                                axis=1)
        pred_bboxes = pred_bboxes.int()

        pred_bboxes[:, 0] = torch.clamp(pred_bboxes[:, 0], min=0)
        pred_bboxes[:, 1] = torch.clamp(pred_bboxes[:, 1], min=0)
        pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2],
                                        max=self.image_w - 1)
        pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3],
                                        max=self.image_h - 1)

        # pred bboxes shape:[points_num,4]
        return pred_bboxes

