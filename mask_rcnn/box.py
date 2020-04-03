import math
import torch
import numpy as np
from torchvision import ops

CAN_USE_GPU = torch.cuda.is_available()


def base_anchor(w, ratios=(0.5, 1, 2)):
    return [((w-1)/2, (w-1)/2, w/math.sqrt(ratio), w*math.sqrt(ratio)) for ratio in ratios]


def box2point(boxes):
    x, y, w, h = (boxes[:, 0]+boxes[:, 2])/2, (boxes[:, 1]+boxes[:, 3])/2, boxes[:, 2]-boxes[:, 0]+1, boxes[:, 3]-boxes[:, 1]+1
    return x, y, w, h


# 把一批x, y, w, h的box转成xmin, ymin, xmax, ymax的box
def to_box(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.round(torch.stack([x-(w-1)/2, y-(h-1)/2, x+(w-1)/2, y+(h-1)/2], dim=0).t())


def build_anchors(f_w, f_h, f_s, ratios=(0.5, 1, 2), scalars=(8, 16, 24)):
    base = torch.Tensor(base_anchor(f_s, ratios=ratios))
    base_expand = to_box(torch.cat([torch.cat([base[:, 0:2], base[:, 2:4]*scalar], dim=1) for scalar in scalars], 0))
    shift_x, shift_y = np.meshgrid(np.array(range(f_w))*f_s, np.array(range(f_h))*f_s)
    shifts = torch.Tensor(np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose())
    anchors = torch.cat([base_expand+shift.view(1, 4) for shift in shifts], dim=0)
    if CAN_USE_GPU:
        anchors = anchors.cuda()
    return anchors


def nms(boxes, sorce, thresh=0.5):
    return ops.nms(boxes, sorce, thresh)


def batch_iou(boxes1, boxes2):
    x1, y1, x2, y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_, y1_, x2_, y2_ = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    lx, ly = torch.stack([x1, x1_], 0).max(dim=0)[0], torch.stack([y1, y1_], 0).max(dim=0)[0]
    rx, ry = torch.stack([x2, x2_], 0).min(dim=0)[0], torch.stack([y2, y2_], 0).min(dim=0)[0]
    mask = ((lx < rx) & (ly < ry)).int()
    jiao_ji = (ry-ly+1)*(rx-lx+1)
    bing_ji = (x2-x1+1)*(y2-y1+1)+(x2_-x1_+1)*(y2_-y1_+1)-jiao_ji
    bing_ji[bing_ji == 0] = 1e-5
    return (jiao_ji/bing_ji)*mask


def batch_boxes_gts_iou(boxes, gts):
    batch_size, box_num, gt_num = gts.shape[0], boxes.shape[1], gts.shape[1]
    return batch_iou(
        boxes.contiguous().view(batch_size, box_num, 1, 4)
            .expand_as(torch.empty(size=(batch_size, box_num, gt_num, 4)))
            .contiguous().view(-1, 4),
        gts.view(batch_size, 1, gt_num, 4)
            .expand_as(torch.empty(size=(batch_size, box_num, gt_num, 4)))
            .contiguous().view(-1, 4)
    ).view(batch_size, box_num, gt_num)


def adjust_anchors(anchors, offset_pred):
    # tx = (x_gt - x_anchor) / w_anchor
    # ty = (y_gt - y_anchor) / h_anchor
    # tw = log(w_gt / w_anchor)
    # th = log(h_gt / h_anchor)
    x_anchor, y_anchor, w_anchor, h_anchor = box2point(anchors)
    tx, ty, tw, th = offset_pred[:, 0], offset_pred[:, 1], offset_pred[:, 2], offset_pred[:, 3]
    return to_box(
        torch.stack([tx*w_anchor+x_anchor, ty*h_anchor+y_anchor,
                     torch.exp(tw)*w_anchor, torch.exp(th)*h_anchor], dim=0).t()
    )


def offset_real(anchors, gts):
    x_anchor, y_anchor, w_anchor, h_anchor = box2point(anchors)
    x_gt, y_gt, w_gt, h_gt = box2point(gts)
    offset_target = torch.stack([(x_gt-x_anchor)/w_anchor, (y_gt-y_anchor)/h_anchor,
                                 torch.log(w_gt/w_anchor), torch.log(h_gt/h_anchor)], dim=0).t()
    if CAN_USE_GPU:
        offset_target = offset_target.cuda()
    return offset_target


def gt_mask_from_gts(gts):
    # gt_mask表达的是一个batch中哪些gt有用，哪些没用, 比如每张图有两个gt，一个batch3张图
    # [ [0, 1]
    #   [1, 0],
    #   [1, 1]] 这样来表示每张图上的gt哪个有用, 没用的gt就是[0, 0, 0, 0]这种填充上去的
    invalid_gt = torch.Tensor([0, 0, 0, 0])
    if CAN_USE_GPU:
        invalid_gt = invalid_gt.cuda()
    gt_mask = torch.stack(
        [torch.Tensor([0 if every_gt.eq(invalid_gt).sum().item() == 4 else 1
                       for every_gt in gt]) for gt in gts], dim=0)
    if CAN_USE_GPU:
        gt_mask = gt_mask.cuda()
    return gt_mask


if __name__ == '__main__':
    boxes = [
        [0, 0, 15, 15],
        [0, 0, 16, 16]
    ]
    sorce = [0.7, 0.8]
    boxes, sorce = torch.Tensor(boxes).cuda(), torch.Tensor(sorce).cuda()
    valid_box_index = nms(boxes, sorce)
    print(valid_box_index)