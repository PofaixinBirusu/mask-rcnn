from torch import nn
from torchvision import ops
import torch
import numpy as np
from mask_rcnn.box import offset_real
from mask_rcnn.box import adjust_anchors
from mask_rcnn.box import nms
from mask_rcnn.util import one_hot
import cv2

CAN_USE_GPU = torch.cuda.is_available()


class RCNN(nn.Module):
    def __init__(self, n_class=2):
        super(RCNN, self).__init__()
        self.n_class = n_class
        # roi网络
        # 预测每个roi的分类
        self.roi_cls_pred_net = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(True),
            nn.Linear(1024, n_class+1)
        )
        # 预测每个roi的边框偏移
        self.roi_offset_pred_net = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4)
        )
        # mask网络
        self.mask_net = nn.Sequential(
            # pos_roi_num x 512 x 7 x 7 输入
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, n_class, kernel_size=1, stride=1),
            nn.Sigmoid()
            # pos_roi_num x n_class x 14 x 14 输出
        )

    # 这个roi是一张图上的n个roi, 比如一张图60个roi, 就是60 x 512 x 7 x 7
    def predict(self, feature_map, regions):
        roi = ops.roi_align(feature_map, [regions], output_size=(7, 7), spatial_scale=1/16)
        roi = roi.view(regions.shape[0], -1)
        cls_pred = self.roi_cls_pred_net(roi)
        offset_pred = self.roi_offset_pred_net(roi)
        return offset_pred, cls_pred

    def mask_predict(self, feature_map, pos_regions):
        roi = ops.roi_align(feature_map, [pos_regions], output_size=(7, 7), spatial_scale=1/16)
        mask_pred = self.mask_net(roi)
        return mask_pred

    def mask_target(self, mask_label, pos_regions, pos_cls, dsize=(14, 14)):
        # mask_label 是一个单通道的cv图片, 也就是h x w
        region_num = pos_regions.shape[0]
        h, w = mask_label.shape[0], mask_label.shape[1]
        pos_regions = pos_regions.cpu()
        target = []
        for box in pos_regions:
            box = box.int()
            xmin, ymin, xmax, ymax = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(xmax, w-1), min(ymax, h-1)
            mask_regions = mask_label[ymin:ymax+1, xmin:xmax+1]
            mask_regions = cv2.resize(mask_regions, dsize)
            # print(mask_regions)
            target.append(list(mask_regions))
        # pos_num x n_class x 14 x 14  这里n_clsss+1是因为标签上背景是0
        target = one_hot(np.array(target), self.n_class+1)
        # 每个类只产生他对应那个类的位置的损失
        single_class_target = []
        for i, cls in enumerate(pos_cls):
            single_class_target.append(torch.Tensor(target[i, cls.item()+1, :, :]))
        return torch.stack(single_class_target, dim=0)

    def mask_forward(self, feature_map, pos_regions, pos_cls, mask_label, dsize=(14, 14)):
        # pos_num x n_class x 14 x 14
        mask_pred = self.mask_predict(feature_map, pos_regions).cpu()
        single_class_target = []
        for i, cls in enumerate(pos_cls):
            single_class_target.append(mask_pred[i, cls.item(), :, :])
        mask_predict = torch.stack(single_class_target, dim=0)
        mask_target = self.mask_target(mask_label, pos_regions, pos_cls, dsize)
        return mask_predict, mask_target

    def loss(self, feature_map, regions, gts, cls, mask_label, sample_num=60):
        gts, cls, mask_label = gts[0], cls[0], mask_label[0]
        mask_label = mask_label.numpy().astype(np.uint8)
        iou = ops.box_iou(regions, gts)
        max_iou, max_iou_gt_index = iou.max(1)
        label = torch.zeros(size=(regions.shape[0], )).fill_(-1)
        label[max_iou >= 0.5] = 1
        label[max_iou < 0.5] = 0
        current_pos_num, target_pos_num = (label == 1).sum().item(), sample_num//4
        if current_pos_num > target_pos_num:
            discard = (label == 1).nonzero().view(-1)[
                torch.Tensor(np.random.permutation(current_pos_num)[:current_pos_num-target_pos_num]).long()
            ]
            label[discard] = -1
        current_neg_num, target_neg_num = (label == 0).sum().item(), sample_num-(label == 1).sum().item()
        if current_neg_num > target_neg_num:
            discard = (label == 0).nonzero().view(-1)[
                torch.Tensor(np.random.permutation(current_neg_num)[:current_neg_num-target_neg_num]).long()
            ]
            label[discard] = -1
        pos_region = regions[label == 1]
        neg_region = regions[label == 0]
        print("pos: %d  neg: %d" % (pos_region.shape[0], neg_region.shape[0]))
        offset_pred, cls_pred = self.predict(feature_map, torch.cat([pos_region, neg_region], dim=0))

        offset_loss_fn, cls_loss_fn = nn.SmoothL1Loss(reduction="sum"), nn.CrossEntropyLoss()
        # 计算roi loss
        pos_gt = gts[max_iou_gt_index[label == 1]]
        # 偏移loss
        offset_target = offset_real(pos_region, pos_gt)
        offset_predict = offset_pred[:offset_target.shape[0]]
        # print(offset_pred.shape)
        # print(offset_target.shape)
        offset_loss = offset_loss_fn(offset_predict, offset_target)/sample_num
        # 分类loss
        pos_cls_target = cls[max_iou_gt_index[label == 1]]
        neg_cls_target = torch.zeros(size=((label == 0).sum().item(), ))+self.n_class
        neg_cls_target = neg_cls_target.long()
        if CAN_USE_GPU:
            neg_cls_target = neg_cls_target.cuda()
        print(cls_pred[:pos_cls_target.shape[0]].argmax(1))
        cls_target = torch.cat([pos_cls_target, neg_cls_target], dim=0)
        cls_loss = cls_loss_fn(cls_pred, cls_target)
        # mask loss
        mask_loss_fn = nn.BCELoss()
        mask_pred, mask_target = \
            self.mask_forward(feature_map, pos_region, cls[max_iou_gt_index[label == 1]], mask_label)
        if CAN_USE_GPU:
            mask_pred, mask_target = mask_pred.cuda(), mask_target.cuda()
        mask_loss = mask_loss_fn(mask_pred, mask_target)
        print("offset loss: %.2f  cls loss: %.2f  mask loss: %.2f" % (offset_loss.item(), cls_loss.item(), mask_loss.item()))
        return offset_loss+cls_loss+mask_loss

    def adjust_regions(self, feature_map, regions, thresh=0.5):
        offset_pred, cls_pred = self.predict(feature_map, regions)
        box_pred, class_index = adjust_anchors(regions, offset_pred), cls_pred.argmax(dim=1)
        box_pred, cls_pred, class_index = \
            box_pred[class_index != self.n_class], cls_pred[class_index != self.n_class], class_index[class_index != self.n_class]

        select_index = nms(box_pred, cls_pred.max(1)[0], thresh=thresh)
        boxes, cls_index, confidence = box_pred[select_index], class_index[select_index], torch.softmax(cls_pred, 1).max(1)[0][select_index]
        mask_pred = self.mask_predict(feature_map, boxes).cpu()
        mask = []
        for i, cls in enumerate(cls_index):
            xmin, ymin, xmax, ymax = boxes[i][0].item(), boxes[i][1].item(), boxes[i][2].item(), boxes[i][3].item()
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            w, h = int(boxes[i][2].item()-boxes[i][0].item()+1), int(boxes[i][3].item()-boxes[i][1].item()+1)
            mask.append(cv2.resize(torch.round(mask_pred[i, cls.item(), :, :]).numpy().astype(np.uint8), dsize=(w, h)))
        return boxes, cls_index, confidence, mask