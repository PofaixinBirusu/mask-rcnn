import torch
from torch import nn
import numpy as np
from mask_rcnn.box import build_anchors
from mask_rcnn.box import batch_boxes_gts_iou
from mask_rcnn.box import gt_mask_from_gts
from mask_rcnn.box import offset_real
from mask_rcnn.box import adjust_anchors
from mask_rcnn.box import nms

CAN_USE_GPU = torch.cuda.is_available()


class RPN(nn.Module):
    def __init__(self, f_w, f_h, f_s=16, scalars=(8, 16, 32)):
        super(RPN, self).__init__()
        # rpn网络
        self.more_deep = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.cls_pred_net = nn.Conv2d(512, 18, kernel_size=1, stride=1)
        self.offset_pred_net = nn.Conv2d(512, 36, kernel_size=1, stride=1)
        self.anchors = build_anchors(f_w, f_h, f_s, scalars=scalars)
        self.anchor_num = self.anchors.shape[0]

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def predict(self, feature_map):
        batch_size = feature_map.shape[0]
        out = self.more_deep(feature_map)
        cls_pred = self.cls_pred_net(out).permute([0, 2, 3, 1])\
            .contiguous().view(batch_size, -1, 18).view(batch_size, -1, 2)
        offset_pred = self.offset_pred_net(out).permute(([0, 2, 3, 1]))\
            .contiguous().view(batch_size, -1, 36).view(batch_size, -1, 4)
        # size: batch_size x wh9 x 4,  batch_size x wh9 x 2
        return offset_pred, cls_pred

    def loss(self, feature_map, gts, sample_num=256):
        batch_size, gt_num = gts.shape[0], gts.shape[1]
        iou = batch_boxes_gts_iou(
            self.anchors.view(1, self.anchor_num, 4)
                .expand_as(torch.empty(batch_size, self.anchor_num, 4)), gts)
        gt_mask = gt_mask_from_gts(gts)
        # batch_size x anchor_num
        max_iou, max_iou_gt_index = iou.max(dim=2)
        label = torch.zeros(batch_size, self.anchor_num).fill_(-1)
        if CAN_USE_GPU:
            label = label.cuda()
        label[max_iou < 0.3] = 0
        # 与每个gt有最大iou的anchor的index batch_size x gt_num
        _, max_iou_anchor_index = iou.permute([0, 2, 1]).max(2)
        for i in range(batch_size):
            label[i, max_iou_anchor_index[i][gt_mask[i] == 1]] = 1
        label[max_iou >= 0.7] = 1
        for i in range(batch_size):
            pos_num, current_pos_num = sample_num//2, (label[i] == 1).sum().item()
            # 正样本太多了，去掉一些
            if current_pos_num > pos_num:
                discard = (label[i] == 1).nonzero().view(-1)[
                    torch.Tensor(np.random.permutation(current_pos_num)[:current_pos_num-pos_num]).long()
                ]
                label[i, discard] = -1
            neg_num, current_neg_num = sample_num-(label[i] == 1).sum().item(), (label[i] == 0).sum().item()
            # 负样本太多
            if current_neg_num > neg_num:
                discard = (label[i] == 0).nonzero().view(-1)[
                    torch.Tensor(np.random.permutation(current_neg_num)[:current_neg_num - neg_num]).long()
                ]
                label[i, discard] = -1
        # 正负样本筛选好了，可以计算损失了
        offset_pred, cls_pred = self.predict(feature_map)
        rpn_loss = 0
        offset_loss_fn, cls_loss_fn = nn.SmoothL1Loss(reduction="sum"), nn.CrossEntropyLoss()
        for i in range(batch_size):
            # 偏移损失
            pos_anchor = self.anchors[label[i] == 1]
            pos_gt = gts[i, max_iou_gt_index[i][label[i] == 1]]
            offset_predict = offset_pred[i][label[i] == 1]
            offset_target = offset_real(pos_anchor, pos_gt)
            offset_loss = offset_loss_fn(offset_predict, offset_target)/sample_num
            # 前景背景分类损失
            pos_cls = cls_pred[i][label[i] == 1]
            pos_target = torch.ones(pos_cls.shape[0]) if not CAN_USE_GPU else torch.ones(pos_cls.shape[0]).cuda()
            neg_cls = cls_pred[i][label[i] == 0]
            neg_target = torch.zeros(neg_cls.shape[0]) if not CAN_USE_GPU else torch.zeros(neg_cls.shape[0]).cuda()

            cls_loss = cls_loss_fn(torch.cat([pos_cls, neg_cls], dim=0), torch.cat([pos_target, neg_target], dim=0).long())
            rpn_loss += cls_loss+offset_loss
        return rpn_loss

    def region_propose(self, feature_map, k1=500, k2=200):
        # 单张图的 batch_size = 1
        offset_pred, cls_pred = self.predict(feature_map)
        offset_pred, cls_pred = offset_pred[0], cls_pred[0]
        # 前景概率作为分数
        cls_sorce = torch.softmax(cls_pred, dim=1)[:, 1]
        top_k_index = torch.topk(cls_sorce, k1, dim=0)[1]
        cls_sorce = cls_sorce[top_k_index]
        select_anchor = self.anchors[top_k_index]
        select_offset = offset_pred[top_k_index]
        box_pred = adjust_anchors(select_anchor, select_offset)
        valid_box_index = nms(box_pred, cls_sorce, thresh=0.6)
        # topk1做完nms后，选topk2
        box_pred = box_pred[valid_box_index][:k2]
        return box_pred